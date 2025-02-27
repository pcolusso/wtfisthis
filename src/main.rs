use anyhow::{Context, Result};
use git2::{Repository, TreeWalkMode};
use polars::prelude::*;
use rust_bert::pipelines::sentence_embeddings::*;
use soa_derive::StructOfArray;
use std::fs;
use std::path::{Path, PathBuf};
use tree_sitter::{Parser, Query, QueryCursor, StreamingIterator};

fn find_files(repo_path: impl AsRef<Path>) -> Result<Vec<PathBuf>> {
    let repo = Repository::open(repo_path).context("Repo doesn't exist")?;

    let mut ruby_files = Vec::new();

    let head = repo.head()?;
    if let Ok(commit) = head.peel_to_commit() {
        let tree = commit.tree()?;

        // Walk through the repository tree
        tree.walk(TreeWalkMode::PreOrder, |dir, entry| {
            if let Some(name) = entry.name() {
                if name.ends_with(".rb") {
                    let file_path = format!("{}{}", dir, name);
                    ruby_files.push(PathBuf::from(file_path));
                }
            }
            0 // Continue walking
        })?;
    }

    Ok(ruby_files)
}

// TODO: This is largely generated, need to better pull this apart and understand wtf is going on
fn chunk_methods(base_path: impl AsRef<Path>, ruby_files: Vec<PathBuf>) -> Result<CodeFragVec> {
    let mut methods = CodeFragVec::new();

    // Initialize tree-sitter parser with Ruby language
    let mut parser = Parser::new();
    let language = tree_sitter_ruby::LANGUAGE;
    parser.set_language(&language.into())?;

    // Create a query to find all method definitions
    let query_str = r#"
        (method
          name: (_) @method_name
          parameters: (_)?
          body: (_)?) @method

        (singleton_method
          object: (_)
          name: (_) @method_name
          parameters: (_)?
          body: (_)?) @method
    "#;

    let query = Query::new(&language.into(), query_str)?;
    let method_capture_index = query.capture_index_for_name("method").unwrap();
    let method_name_index = query.capture_index_for_name("method_name").unwrap();

    for path in ruby_files {
        let path = base_path.as_ref().join(&path);
        //println!("Processing: {}", path.display());

        let content = fs::read_to_string(&path)
            .context(format!("Failed to read file: {}", path.display()))?;
        let content_bytes = content.as_bytes();

        // Parse the file
        let tree = parser
            .parse(&content, None)
            .ok_or(anyhow::anyhow!("Failed to parse file"))?;

        // Get the root node
        let root_node = tree.root_node();
        let mut query_cursor = QueryCursor::new();
        let mut matches = query_cursor.matches(&query, root_node, content_bytes);

        while let Some(match_result) = matches.next() {
            // Find the method node and name
            if let Some(method_capture) = match_result
                .captures
                .iter()
                .find(|c| c.index == method_capture_index)
            {
                let method_node = method_capture.node;
                let method_name = match_result
                    .captures
                    .iter()
                    .find(|c| c.index == method_name_index)
                    .map(|c| c.node.utf8_text(content_bytes))
                    .transpose()?
                    .unwrap_or("unknown");

                // Get the method text directly
                let method_text = method_node.utf8_text(content_bytes)?.to_string();

                // Get line number (tree-sitter uses 0-based indexing)
                let line_number = method_node.start_position().row + 1;

                println!("Found method '{}' at line {}", method_name, line_number);

                let frag = CodeFrag {
                    path: path
                        .clone()
                        .into_os_string()
                        .into_string()
                        .expect("Failed to convert path to string"),
                    line_number,
                    body: method_text,
                    embeds: None,
                };

                methods.push(frag);
            }
        }
    }

    Ok(methods)
}

#[derive(StructOfArray)]
struct CodeFrag {
    body: String,
    path: String,
    line_number: usize,
    embeds: Option<Vec<f32>>,
}

impl std::fmt::Debug for CodeFrag {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CodeFrag")
            .field("loc", &format!("{}:{}", self.path, self.line_number))
            .field(
                "body",
                &format!(
                    "Lines: {}, Signature: {:?}",
                    self.body.lines().count(),
                    self.body.lines().take(1)
                ),
            )
            .field("embeds", &self.embeds)
            .finish()
    }
}

fn create_embeddings(code_frags: &mut CodeFragVec) -> Result<()> {
    let model = SentenceEmbeddingsBuilder::remote(SentenceEmbeddingsModelType::AllMiniLmL12V2)
        .create_model()?;

    // 🤞 the order is preserved!
    let res = model.encode(&code_frags.body)?;
    code_frags.embeds = res.into_iter().map(|embed| Some(embed)).collect();
    Ok(())
}

fn code_frag_to_dataframe(code_frag_vec: CodeFragVec) -> Result<DataFrame> {
    let body_series = Series::new("body".into(), &code_frag_vec.body);
    let path_series = Series::new("path".into(), &code_frag_vec.path);
    let line_number_u32: Vec<u32> = code_frag_vec
        .line_number
        .iter()
        .map(|x| *x as u32)
        .collect();
    let line_number_series = Series::new("line_number".into(), &line_number_u32);

    // TODO: cleanup this mess
    let unwrapped_embeds: Vec<Vec<f32>> = code_frag_vec
        .embeds
        .iter()
        .map(|embed| embed.as_ref().unwrap().clone())
        .collect();

    let x: Vec<AnyValue> = unwrapped_embeds
        .into_iter()
        .map(|v| AnyValue::List(Series::new("".into(), v)))
        .collect();

    let embeds_series = Series::new("embeds".into(), x);

    // Create the DataFrame
    let df = DataFrame::new(vec![
        body_series.into(),
        path_series.into(),
        line_number_series.into(),
        embeds_series.into(),
    ])?;

    Ok(df)
}

fn write_dataframe_to_parquet(df: &mut DataFrame, path: impl AsRef<Path>) -> Result<()> {
    let dest = path.as_ref().join("embeddings.parquet");
    let file = fs::File::create(dest)?;
    let writer = ParquetWriter::new(file).with_compression(ParquetCompression::Snappy);
    writer.finish(df)?;
    Ok(())
}

// Also largely generated. I see an issue with going row-wise.
// Can we take out the embeds series, apply dot product, and then sort on that?
// iirc apply can also do parallel stuff too...
fn search(df: &DataFrame, query: &str) -> Result<DataFrame> {
    let model = SentenceEmbeddingsBuilder::remote(SentenceEmbeddingsModelType::AllMiniLmL12V2)
        .create_model()?;

    let query_embeddings = model.encode(&[query])?;
    let query_embedding = &query_embeddings[0];

    let embeds_series = df.column("embeds")?;
    let mut similarities_with_indices: Vec<(usize, f32)> = Vec::with_capacity(df.height());

    for i in 0..df.height() {
        let embedding = match embeds_series.get(i) {
            Ok(AnyValue::List(list)) => {
                // Extract the embedding vector - adjust based on how your embeddings are stored
                let embed_vec: Vec<f32> = list
                    .iter()
                    .map(|v| match v {
                        AnyValue::Float32(f) => f,
                        _ => 0.0, // Handle type mismatch
                    })
                    .collect();
                embed_vec
            }
            _ => continue, // Skip if can't get a valid embedding
        };

        // Calculate dot product (cosine similarity if vectors are normalized)
        let similarity = query_embedding
            .iter()
            .zip(embedding.iter())
            .map(|(a, b)| a * b)
            .sum();

        similarities_with_indices.push((i, similarity));
    }

    // 5. Sort by similarity (descending)
    similarities_with_indices
        .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    let top_indices: Vec<u32> = similarities_with_indices
        .iter()
        .take(5)
        .map(|(idx, _)| *idx as u32)
        .collect();

    let paths = df.column("path")?.str()?;
    let line_numbers = df.column("line_number")?.u32()?;
    let bodies = df.column("body")?.str()?;

    for idx in top_indices.iter() {
        let line_number = line_numbers.get(*idx as usize).unwrap();
        let path = paths.get(*idx as usize).unwrap();
        let body = bodies.get(*idx as usize).unwrap();
        println!("Found in {}:{} -> \n{}\n---\n", path, line_number, body);
    }

    let x = IdxCa::new("".into(), top_indices);

    let mask = df.take(&x)?;

    Ok(mask)
}

fn scan(path: impl AsRef<Path>) -> Result<DataFrame> {
    let files = find_files(path.as_ref())?;
    let mut frags = chunk_methods(path.as_ref(), files)?;
    create_embeddings(&mut frags)?;

    let mut df = code_frag_to_dataframe(frags)?;
    write_dataframe_to_parquet(&mut df, path.as_ref())?;

    Ok(df)
}

fn load(path: impl AsRef<Path>) -> Result<DataFrame> {
    let p = path.as_ref().join("embeddings.parquet");
    // let p = Path::new("embeddings.parquet");
    Ok(ParquetReader::new(fs::File::open(p).context("No embeddings to load.")?).finish()?)
}

#[derive(clap::Parser)]
struct Opts {
    #[command(subcommand)]
    action: Action,
    repo: PathBuf,
    query: String
}

#[derive(clap::Subcommand, Clone)]
enum Action {
    Scan,
    Load,
}

fn main() -> Result<()> {
    use clap::Parser;
    let Opts{ repo, action, query }  = Opts::parse();
    let df = match action {
        Action::Scan => scan(repo)?,
        Action::Load => load(repo)?,
    };

    search(&df, &query)?;

    Ok(())
}

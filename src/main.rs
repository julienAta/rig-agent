mod agent;
use std::io::{self, Write};
use dotenv::dotenv;
use mongodb::{
    bson::{self, doc},
    options::ClientOptions,
    Client as MongoClient, Collection,
};
use rig::providers::openai::{self, TEXT_EMBEDDING_ADA_002};
use rig::embeddings::EmbeddingsBuilder;
use rig_mongodb::{MongoDbVectorIndex, SearchParams};
use serde::Deserialize;
use anyhow::Result;

// Structure pour stocker nos connaissances
#[derive(Embed, Clone, Deserialize, Debug)]
struct Knowledge {
    #[serde(rename = "_id")]
    id: String,
    #[embed]
    definition: String,
}

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    dotenv().ok();

    // Initialize OpenAI client
    let openai_client = openai::Client::from_env();
    let ada = agent::Ada::new()?;

    // Initialize MongoDB client
    let mongodb_uri = std::env::var("MONGODB_CONNECTION_STRING").expect("MONGODB_CONNECTION_STRING not set");
    let options = ClientOptions::parse(mongodb_uri)
        .await
        .expect("MongoDB connection string should be valid");
    let mongodb_client = MongoClient::with_options(options)?;

    // Initialize MongoDB collection
    let collection: Collection<bson::Document> = mongodb_client
        .database("knowledgebase")
        .collection("context");

    // Setup embedding model and index
    let model = openai_client.embedding_model(TEXT_EMBEDDING_ADA_002);
    let index = MongoDbVectorIndex::new(collection.clone(), model.clone(), "vector_index", SearchParams::new()).await?;

    println!("Chat with Ada (type 'exit' to quit, 'learn X' to teach Ada something new)");
    
    loop {
        print!("> ");
        io::stdout().flush()?;
        
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();
        
        if input.eq_ignore_ascii_case("exit") {
            println!("Goodbye!");
            break;
        }

        // Handle learning new information
        if input.starts_with("learn ") {
            let new_knowledge = input[6..].trim();
            let knowledge = Knowledge {
                id: uuid::Uuid::new_v4().to_string(),
                definition: new_knowledge.to_string(),
            };

            let embeddings = EmbeddingsBuilder::new(model.clone())
                .document(knowledge)?
                .build()
                .await?;

            let mongo_document = embeddings
                .iter()
                .map(|(Knowledge { id, definition, .. }, embedding)| {
                    doc! {
                        "id": id.clone(),
                        "definition": definition.clone(),
                        "embedding": embedding.first().vec.clone(),
                    }
                })
                .collect::<Vec<_>>();

            collection.insert_many(mongo_document, None).await?;
            println!("Added new knowledge to memory!");
            continue;
        }

        // Search for relevant context
        let results = index.top_n::<Knowledge>(input, 2).await?;
        let context = if !results.is_empty() {
            let contexts: Vec<String> = results.iter()
                .map(|(_, _, doc)| doc.definition.clone())
                .collect();
            println!("Found relevant context: {:?}", contexts);
            contexts.join("\n")
        } else {
            String::new()
        };

        // Generate response using context
        let prompt = if context.is_empty() {
            input.to_string()
        } else {
            format!("Context:\n{}\n\nQuestion: {}", context, input)
        };

        let response = ada.respond(&prompt).await?;
        println!("Ada: {}", response);
    }
    
    Ok(())
}
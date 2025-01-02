use mongodb::{bson::{self, doc}, Collection};
use rig::providers::openai::{Client, TEXT_EMBEDDING_ADA_002};
use rig::embeddings::EmbeddingsBuilder;
use rig_mongodb::{MongoDbVectorIndex, SearchParams};
use serde::{Deserialize, Serialize};
use anyhow::Result;

#[derive(Embed, Clone, Deserialize, Debug, Serialize)]
pub struct Knowledge {
    #[serde(rename = "_id")]
    id: String,
    #[embed]
    content: String,
}

pub struct KnowledgeStore {
    collection: Collection<bson::Document>,
    index: MongoDbVectorIndex,
}

impl KnowledgeStore {
    pub async fn new(collection: Collection<bson::Document>, openai_client: &Client) -> Result<Self> {
        let model = openai_client.embedding_model(TEXT_EMBEDDING_ADA_002);
        
        let index = MongoDbVectorIndex::new(
            collection.clone(),
            model,
            "vector_index",
            SearchParams::new()
        ).await?;

        Ok(Self {
            collection,
            index,
        })
    }

    pub async fn add_content(&self, content: &str) -> Result<()> {
        let knowledge = Knowledge {
            id: uuid::Uuid::new_v4().to_string(),
            content: content.to_string(),
        };

        let embeddings = EmbeddingsBuilder::new(self.index.model().clone())
            .document(knowledge)?
            .build()
            .await?;

        let mongo_documents = embeddings
            .iter()
            .map(|(Knowledge { id, content, .. }, embedding)| {
                doc! {
                    "id": id.clone(),
                    "content": content.clone(),
                    "embedding": embedding.first().vec.clone(),
                }
            })
            .collect::<Vec<_>>();

        self.collection.insert_many(mongo_documents, None).await?;
        Ok(())
    }

    pub async fn search(&self, query: &str, limit: usize) -> Result<Vec<String>> {
        let results = self.index.top_n::<Knowledge>(query, limit).await?;
        Ok(results.into_iter().map(|(_, _, doc)| doc.content).collect())
    }
}
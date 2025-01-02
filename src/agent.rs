use rig::{completion::Prompt, providers::openai, agent::Agent};
use anyhow::Result;

pub struct Ada {
    agent: Agent<openai::CompletionModel>,  // Spécifions le type générique
}

impl Ada {
    pub fn new() -> Result<Self> {
        let openai_client = openai::Client::from_env();
        let agent = openai_client.agent("gpt-4")
            .preamble("You are Ada, an enthusiastic programmer and AI researcher who loves Rust. 
                      You're optimistic about technology and often use programming metaphors in your speech.")
            .build();

        Ok(Self { agent })
    }

    pub async fn respond(&self, input: &str) -> Result<String> {
        let response = self.agent.prompt(input).await?;
        Ok(response)
    }
}
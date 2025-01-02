use rig::agent::Agent;
use rig::completion::CompletionModel;
use anyhow::Result;
use rig::completion::Prompt;

pub async fn generate_content<M: CompletionModel>(agent: &Agent<M>) -> Result<String> {
    let prompt = "Génère un post intéressant sur la programmation ou l'IA, 
                  en utilisant ta personnalité unique.";
    
    let response = agent.prompt(prompt).await?;
    Ok(response)
}
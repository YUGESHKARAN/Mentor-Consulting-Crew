import os
from typing import List
from crewai import Crew, Agent, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task

from crewai_tools import SerperDevTool, ScrapeWebsiteTool, DirectoryReadTool, FileReadTool, FileWriterTool, YoutubeChannelSearchTool
from pydantic import BaseModel, Field
from dotenv import load_dotenv

_ = load_dotenv()  # Load environment variables from .env file

llm = LLM(
model="gemini/gemini-2.0-flash",
temperature=0.7,
api_key=os.getenv("Google_API_KEY")
)

class Content(BaseModel):
    day_or_week_number: int = Field(..., description="The day or week number of the content")
    task_title: str = Field(..., description="The day wise title of the content")
    summary: str = Field(..., description="A brief summary of the content")
    video_url: str = Field(..., description=" The youtube or other source video URL")
    github_url: str = Field(..., description="The GitHub repository URL")
    task_level: str = Field(..., description="The difficulty level of the task")


@CrewBase
class TheConsultantCrew():
    """A crew to create day wise content for a programming course."""
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    @agent
    def research_development(self)-> Agent:
         return Agent(
            config=self.agents_config['research_development'],
            tools=[
                SerperDevTool(),
                ScrapeWebsiteTool(),
                DirectoryReadTool('resources/draft2'),
                FileWriterTool(),
                FileReadTool()
            ],
            reasoning=True,
            inject_date=True,
            llm=llm,
            allow_delegation=True,
            max_rpm=1
        )
    
    @agent
    def content_creator(self)-> Agent:
        return Agent(
            config=self.agents_config['content_creator'],
            tools=[
                # SerperDevTool(),
                ScrapeWebsiteTool(),
                # GithubSearchTool(),
                YoutubeChannelSearchTool(
                     config=dict(
                        llm=dict(
                            provider="google", # or google, openai, anthropic, llama2, ...
                            config=dict(
                                model="gemini/gemini-2.0-flash",
                                # temperature=0.5,
                                # top_p=1,
                                # stream=true,
                            ),
                        ),
                        embedder=dict(
                            provider="google", # or openai, ollama, ...
                            config=dict(
                                model="models/embedding-001",
                                # task_type="retrieval_document",
                                # title="Embeddings",
                            ),
                        ),
                    )
                ),
                DirectoryReadTool('resources/draft2'),
                FileWriterTool(),
                FileReadTool()
            ],
            reasoning=True,
            inject_date=True,
            llm=llm,
            allow_delegation=True,
            max_rpm=1
        )
  
    @agent
    def tasks_scheduler(self)-> Agent:
         return Agent(
            config=self.agents_config['tasks_scheduler'],
            tools=[
                # SerperDevTool(),
                # ScrapeWebsiteTool(),
                DirectoryReadTool('resources/draft2'),
                FileWriterTool(),
                FileReadTool()
            ],
            reasoning=True,
            inject_date=True,
            llm=llm,
            allow_delegation=True,
            max_rpm=1
        )
    
    @task
    def research_consultant(self)-> Task:
        return Task(
            config=self.tasks_config['research_consultant'],
            agent=self.research_development(),
        )
    
    @task
    def path_planner(self)-> Task:
        return Task(
            config=self.tasks_config['path_planner'],
            agent=self.research_development(),
        )
    
    @task
    def content_reviewer(self)-> Task:
        return Task(
            config=self.tasks_config['content_reviewer'],
            agent=self.content_creator(),
        )
    @task
    def create_content_calendar(self)-> Task:
        return Task(
            config=self.tasks_config['create_content_calendar'],
            agent=self.tasks_scheduler(),
        )

    @crew
    def consultantcrew(self) -> Crew:
        """A crew designed to create day-wise learning content, covering a specified technology from scratch to advanced level."""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
            planning=True,
            planning_llm=llm,
            max_rpm=1
        )

if __name__ == "__main__":
    from datetime import datetime

    inputs = {
        "domain_name": "Data Science",
        "topic": "Deep Learning",
        "learning_level": "beginner to advanced",
        "preferred_language": "English",
        "estimated_time": "2 months",
        "current_date": datetime.now().strftime("%Y-%m-%d")
    }


    crew = TheConsultantCrew()
    crew.consultantcrew().kickoff(inputs=inputs)
    print("Crew process completed.")
    
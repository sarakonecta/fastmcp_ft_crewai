from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.mcp import MCPServerStdio
from typing import List
from dotenv import load_dotenv
import os

env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '.env')
load_dotenv(dotenv_path=env_path, override=True)

@CrewBase
class DiceRoller():
    """DiceRoller crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    def __init__(self):
        """Initialize LLM and MCP configuration"""
        self.llm_model = LLM(
            model=os.getenv("MODEL", "openai/gemini-2.5-flash"),
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_API_BASE")
        )
        self.mcps = [
            # Local FastMCP server using stdio transport
            MCPServerStdio(
                command="python",
                args=["/home/ssansalvador/Projects/fastmcp_server/DiceRoller_mcp/server.py"],
                env=None,
            ),
        ]
    
    @agent
    def gambler(self) -> Agent:
        return Agent(
            config=self.agents_config['gambler'], # type: ignore[index]
            verbose=True,
            llm=self.llm_model,
            mcps=self.mcps
        )
        
    @task
    def dice_roll(self) -> Task:
        return Task(
            config=self.tasks_config['dice_roll'], # type: ignore[index]
            output_file='report.md'
        )

    @crew
    def crew(self) -> Crew:
        """Creates the DiceRoller crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks,   # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
        )

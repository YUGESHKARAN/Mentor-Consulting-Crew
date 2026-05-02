from datetime import datetime
from crew import CrewBase
from crew import TheConsultantCrew, Content

def run_crew(inputs: dict):
    """
    Run the crew with dynamic inputs
    """
    # Validate inputs against your BaseModel
    content_inputs = Content(**inputs)
    
    # Create crew instance
    crew_instance = TheConsultantCrew()
    
    # Execute crew with validated inputs
    result = crew_instance.consultantcrew().kickoff(inputs=content_inputs.dict())
    
    return result

if __name__ == "__main__":
    # For local testing - this will be replaced by API inputs in deployment
    test_inputs = {
        "domain_name": "Data Science",
        "topic": "Deep Learning", 
        "learning_level": "beginner to advanced",
        "preferred_language": "English",
        "estimated_time": "2 months",
        "current_date": datetime.now().strftime("%Y-%m-%d")
    }
    
    result = run_crew(test_inputs)
    print("Crew execution completed:", result)
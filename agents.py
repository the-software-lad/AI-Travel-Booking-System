import asyncio
import json
import re
from typing import Any 
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.types import interrupt, Command
from config import get_llm
from mcp_client import current_weather, forecast, list_airlines, list_airports, tavily_search
from state import TravelState

llm = get_llm()

def _llm_text(system: str, prompt: str, **kwargs):
    response = llm.invoke(
        [
            SystemMessage(content=system),
            HumanMessage(content=prompt)
        ],
        **kwargs
    )
    usage = getattr(response, "usage_metadata", None)
    if usage:
        print(f"\nLLM Token Usage: {usage}")
        
    content = response.content
    # Ensure content is always a string even if the model/LangChain returns a list
    if isinstance(content, list):
        content = "".join([item.get("text", str(item)) if isinstance(item, dict) else str(item) for item in content])
    return str(content)

def _json_from_llm(text: Any) -> dict:
    # Handle cases where text is passed as a list
    if isinstance(text, list):
        text = "".join([item.get("text", str(item)) if isinstance(item, dict) else str(item) for item in text])
    elif not isinstance(text, str):
        text = str(text)


    if not text.strip():
        raise ValueError("Received empty response from LLM.")
    
    # Strip markdown code blocks if the LLM wrapped the JSON
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned)
        cleaned = cleaned.strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    
    if start == -1 or end == -1 or start > end:
        raise ValueError(f"LLM did not return a valid JSON object. Raw response was:\n{text}")

    json_text = cleaned[start:end+1]


    return json.loads(json_text)

def supervisor_agent(state: TravelState):
    query = state.get("user_query", "")
    
    if not query:
        # Fallback if query is missing from state
        query = "Plan a trip based on previous context."

    guardrail_prompt = f"""
        Determine whether the following request is a valid travel planning request.

        Return only JSON in this format
        {{
            "allowed": true,
            "reason": ""
        }}

        User Request : {query}
    """

    guardrail_raw = _llm_text("You are an input validation guardrail. Return strict JSON only.", guardrail_prompt)

   

    guardrail_result = _json_from_llm(guardrail_raw)

    
    if not guardrail_result.get("allowed",False):
        reason = guardrail_result.get("reason"," Request rejected by input guardrail")

        return{
            "selected_agents":[],
            "trip_constraints":{},
            "supervisor_reasoning":reason,
            "final_response": reason,
            "messages":[AIMessage(content=f"Guardrail blocked request : {reason}")],
            "llm_calls": state.get("llm_calls",0) + 1
        }




    prompt = f"""
        You are a supervisor of a real world multi agent travel planning system.
        Decide which specialist agents are needed for this user request.

        Available Agents:
        flight_agent: use when flights, airports, airlines, routes or airfare guidance are needed
        hotel_agent: use when hotel, stays, neighbourhoods, or accommodation are needed
        weather_agent: use when weather, climate, season, packing or forecast is needed
        budget_agent: use when budget, affordability, cost or price constraints needed
        itinerary_agent: almost always needed to produce the travel plan

        Return only JSON with this exact schema:
        {{
            "selected_agents":["flight_agent", "hotel_agent", "weather_agent", "budget_agent", "itinerary_agent"],
            "trip_constraints":{{
                "destination":"",
                "origin": "",
                "duration":"",
                "budget":"",
                "travel_style":"",
                "special_preferences":[]
            }},
            "reasoning":""
        }} 

        User Request: {query}
    """

    raw = _llm_text(
        "You route work to specialist agents. Return strict JSON only with no extra commentary.",
        prompt
    )

    parsed = _json_from_llm(raw)
    selected = parsed.get("selected_agents", ["itinerary_agent"])

    return {
        "selected_agents": selected,
        "trip_constraints": parsed.get("trip_constraints", {}),
        "supervisor_reasoning": parsed.get("reasoning", ""),
        "messages": [AIMessage(content="Supervisor created the agent plan.")],
        "llm_calls": state.get("llm_calls", 0) + 1
    }

def flight_agent(state: TravelState):
    query= state["user_query"]
    constraints = state["trip_constraints"]
    destination = constraints["destination"]

    airports = asyncio.run(list_airports(destination, limit=10))
    airlines = asyncio.run(list_airlines("", limit=10))

    prompt = f"""
        Create flight guidance for this trip

        User Request : {query}
        Trip Constraints : {constraints}
        Airport MCP Data : {airports}
        Airline MCP Data : {airlines}

        Return:
        - Relevant departure and arrival airports.
        - Relevant airlines.
        - Estimated duration and fare guidance, if available.
        - One booking recommendation.

        Do not invent flight schedules or confirmed fares.
        Maximum 100 words.
    """
    result = _llm_text(
        "You are a concise flight planning specialist.",
        prompt,
       
    )


    return{
        "flight_results": result,
        "messages":[AIMessage("Flight agent completed.")],
        "llm_calls": state.get("llm_calls", 0) + 1
    }

def hotel_agent(state : TravelState):
    query = f"Best hotels and areas to stay for : {state['user_query']}"

    result = asyncio.run(tavily_search(query))


    return{
        "hotel_results": result,
        "messages":[AIMessage("Hotel agent completed")]
    }

def weather_agent(state: TravelState):
    constraints = state["trip_constraints"]
    city = constraints['destination']

    weather_data = asyncio.run(current_weather(city))
    forecast_data = asyncio.run(forecast(city))

    result = f"""
        Current Weather : {weather_data}

        Forecast : {forecast_data}
    """


    return{
        "weather_results": result,
        "messages" : [AIMessage(content="Weather agent completed")]
    }

def budget_agent(state: TravelState):
    prompt = f"""
        Evaluate this travel budget.

        User request:
        {state.get("user_query")}

        Constraints:
        {state.get("trip_constraints")}

        Flights:
        {state.get("flight_results")}

        Hotels:
        {state.get("hotel_results")}

        Weather:
        {state.get("weather_results")}

        Provide:
        1. Estimated major expenses.
        2. Budget risks.
        3. Money-saving suggestions.
        4. Overall feasibility.

        Maximum 100 words.
        Use bullet points.
        Do not repeat input information.
    """

    result = _llm_text("You are a concise travel budget analyst.", prompt,)

    
    return{
        "budget_results": result,
        "messages": [AIMessage(content="Budget Agent completed")],
        "llm_calls": state.get("llm_calls",0) + 1
    }

def itinerary_agent(state: TravelState):
    prompt = f"""
        create a concise draft travel itinerary.

        User Request : {state["user_query"]}

        Constraints : {state.get("trip_constraints",{})}
        
        Flight Results : {state.get('flight_results',"")}
        
        Hotel Results : {state.get('hotel_results',"")}
        
        Weather Result : {state.get("weather_results","")}

        Budget Results : {state.get("budget_results", "")}

        REQUIREMENTS:
        - Create a day-wise itinerary.
        - Include 3-4 activities per day.
        - Include approximate timings.
        - Respect the available budget.
        - Include essential travel recommendations.
        - Do not invent confirmed bookings.
        - Keep the entire response under 150 words.
        - Use concise bullet points.

        make the output structured, practical and ready fo human review
    """
    result = _llm_text("You are a concise professional travel itinerary planner.", prompt, )


    approval_request = f"""
        Please review this draft travel plan 
        {result}

        Please approval or provide feedback
    """

    return {
        "itinerary": result,
        "approval_request" : approval_request,
        "messages":[AIMessage(content="Draft Itinerary created for Human Review")],
        "llm_calls": state.get("llm_calls",0 )+1
    }

def human_approval_agent(state: TravelState):

    feedback = interrupt({
        "message": "Please review the proposed travel itinerary.",
        "itinerary": state.get("itinerary", ""),
        "budget_notes": state.get("budget_notes", ""),
        "question": "Do you approve this itinerary?"
    })

    # Safely extract the user's response
    if not isinstance(feedback, dict):
        feedback = {}

    approved = feedback.get("approved", False)
    human_feedback = feedback.get("feedback", "")


    return {
        "approved": approved,
        "human_feedback": human_feedback
    }

# def human_approval_agent(state: TravelState):
#     feedback= interrupt(
#         {
#             "question": "do you approve this itinerary?",
#             "draft_itinerary": state.get("itinerary", ""),
#             "approval_request": state.get("approval_request",""),
#             "expected_response": {
#                 "approved": True,
#                 "feedback": "optional feedback for revision"
#             }
#         }
#     )

#     approved= feedback["approved", False]
#     human_feedback = feedback["feedback"]

#     return{
#         "approved": approved,
#         "human_feedback": human_feedback,
#         "messages":[AIMessage(content="Human approval step completed")]
#     }

def final_response_agent(state: TravelState):

    if state["approved"]:
        prompt = f"""
            the human approved this draft itinerary.

            Produce the final polished travel plan.

            Draft Itinerary : {state["itinerary"]}

            Budget Notes : {state["budget_results"]}
        """
    else:
        prompt = f"""
            The human did not approve the draft.

            Original User Request : {state["user_query"]}

            Draft Itinerary: {state["itinerary"]}

            Human Feedback : {state["human_feedback"]}

            budget Notes : {state["budget_results"]}
        """

    result = _llm_text("You are and expert travel, produce final user-ready travel plan.", prompt)


    return{
        "final_response": result,
        "messages": [AIMessage(content=result)],
        "llm_calls": state.get("llm_calls", 0) + 1
    }
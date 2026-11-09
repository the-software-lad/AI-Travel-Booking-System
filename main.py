import os
from typing import TypedDict, Annotated
import operator
import asyncio
import psycopg
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.postgres import PostgresSaver
from langchain_core.messages import (AnyMessage, HumanMessage, AIMessage, SystemMessage)
from langchain_groq import ChatGroq
from mcp_client import tavily_mcp_search, get_airlines, get_airports, aviation_mcp_call, forecast_mcp_search, weather_mcp_search, extract_destination
from psycopg.rows import dict_row

from dotenv import load_dotenv
load_dotenv()

llm = ChatGroq(model= "openai/gpt-oss-20b")

DATABASE_URL = os.getenv("DATABASE_URL")

class TravelState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    user_query: str
    flight_results: str
    hotel_results: str
    itinerary: str
    llm_calls: int
    weather_results : str

FLIGHT_AGENT_PROMPT = """
    You are a travel flight expert.

    User Query: {query}

    Airport Information: {airport_data}

    Airline Information: {airline_data} 

    Generate:
    1.Likely departure airport
    2.Likely arrival airport
    3.Airline serving this route
    4.Typical flight duration
    5.Estimated airfare range
    6.Peak season pricing warning
    7.Booking advice

    Return concise travel guidance
"""

#flight agent
def flight_agent(state: TravelState):
    query = state["user_query"]

    try:
        airports = asyncio.run(
            aviation_mcp_call("list_airports")
        )

        airlines = asyncio.run(
            aviation_mcp_call("list_airlines")
        )

        prompt = FLIGHT_AGENT_PROMPT.format(
            query = query,
            airport_data = str(airports[:3000]),
            airline_data = str(airlines[:3000])
            )

        response = llm.invoke([
            SystemMessage(content="You are an expert travel flight planner."),
            HumanMessage(content=prompt)
        ])

        flight_data = response.content

    except Exception as e :
        flight_data = f" flight information unavailable: {str(e)}"

    return{
            "flight_results" : flight_data,
            "messages" : [
                AIMessage(content="Flight recommendation generated")
            ],
            "llm_calls":state.get("llm_calls",0)+1
        } 


#hotel agent
def hotel_agent(state: TravelState):
    query = f"Best hotels for {state['user_query']}"
    hotel_results = asyncio.run(tavily_mcp_search(query))
    return{
        "hotel_results": hotel_results,
        "messages":[AIMessage(content="Hotel information fetched")]
    }


#itinerary agent
def itinerary_agent(state: TravelState):
    prompt= f"""
            Create a travel itinerary .
            User Query : {state["user_query"]}
            Flight Results : {state["flight_results"]}
            Hotel Results : {state["hotel_results"]}
            Weather Information : {state["weather_results"]}
            """
    response = llm.invoke([SystemMessage(content = "You are an expert Travel Planner"), HumanMessage(content=prompt)])
    return{
        "itinerary" : response.content,
        "messages" : [response],
        "llm_calls" : state.get("llm_calls",0) + 1
    }


#weather agent
def weather_agent(state: TravelState):
    city = extract_destination(state["user_query"])

    weather_data = asyncio.run(weather_mcp_search(city))
    forecast_data = asyncio.run(forecast_mcp_search(city))
    return{
        "weather_results": f"""
            Current Weather : {weather_data}
            Forecast : {forecast_data}
        """,
        "messages":[
            AIMessage(content="Weather details fetched")
        ]
    }


graph = StateGraph(TravelState)

graph.add_node("flight_agent", flight_agent)
graph.add_node("hotel_agent", hotel_agent)
graph.add_node("weather_agent", weather_agent)
graph.add_node("itinerary_agent", itinerary_agent)

graph.add_edge(START, "flight_agent")
graph.add_edge("flight_agent", "hotel_agent")
graph.add_edge("hotel_agent", "weather_agent")
graph.add_edge("weather_agent","itinerary_agent")
graph.add_edge("itinerary_agent", END)

_conn = psycopg.connect(DATABASE_URL, autocommit=True, row_factory=dict_row)
checkpointer = PostgresSaver(_conn)
checkpointer.setup()

app = graph.compile(checkpointer=checkpointer)


if __name__ == "__main__":

    # config={
    #     "configurable":{
    #         "thread_id": "user_shariq1"
    #     }
    # }
    import uuid
    config={
        "configurable":{
            "thread_id":str(uuid.uuid4())
        }
    }

    user_input = input("Enter Travel Request: ")
    result = app.invoke(
        {
            "messages": [
                            HumanMessage(content = user_input)
            ],
            "user_query" : user_input,
            "flight_results": "",
            "hotel_results" : "",
            "itinerary" : "",
            "llm_calls" : 0
        },
        config=config
    )

    print("\n--- Final Response ---\n")

    for msg in result["messages"]:
        print(msg.content)
import os 
from apikey import apikey
import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory

os.environ['OPENAI_API_KEY'] = apikey


st.title('🍝 Meal Planner 🥗')
st.subheader('Plan meals, get recipes, and make shopping lists with OpenAI')

criteria = st.text_input('Enter some criteria for your meal plan. E.g. Vegetarian, high protein, low carb, etc.')
nogo = st.text_input('Enter any foods you don\'t want to eat. E.g. broccoli, tofu, etc.')
days = st.slider('How many days do you want to plan for?', 1, 7, 3)
meals = st.multiselect('What meals do you want to plan?', ['Breakfast', 'Lunch', 'Dinner'])

meals_prompt = """You are an expert chef that can cater to any preference.
You are constructing a {days}-day meal plan for a client that includes {meals}.

Plan the meals around the following criteria: 
{criteria}. 

Avoid using the following foods: 
{nogo}.
"""

meals_template = PromptTemplate(template=meals_prompt, input_variables=['days', 'meals', 'criteria', 'nogo'])

recipes_prompt = """You are an expert chef who excels at writing detailed and clear recipes.
Based on the following meal titles, write out a recipe for each one. List quantities and exact steps. 
Meals:
{meal_output}
"""
recipes_template = PromptTemplate(template=recipes_prompt, input_variables=['meal_output'])

shopping_prompt = """You are a detail oriented personal assistant. You are tasked with preparing a shopping list from the following recipes:
{recipes}

List all of the ingredients and quantities needed, grouping items by similarity for easier shopping. 
Suggest when an ingredient may already be in the pantry.
"""
shopping_template = PromptTemplate(template=shopping_prompt, input_variables=['recipes'])

llm = OpenAI(temperature=0.5)

mealchain = LLMChain(llm=llm, prompt=meals_template, output_key='meal_output')
recipechain = LLMChain(llm=llm, prompt=recipes_template, output_key='recipes')
shoppingchain = LLMChain(llm=llm, prompt=shopping_template, output_key='shopping')

textchain = SequentialChain(chains=[mealchain, recipechain, shoppingchain], 
                            input_variables=['days', 'meals', 'criteria', 'nogo'], 
                            output_variables=['meal_output', 'recipes', 'shopping'], 
                            verbose=True)

if st.button('Generate'):
    if criteria and nogo and days and meals:
        with st.spinner('Generating response...'):
            results = textchain({
                'days': days,
                'meals': meals,
                'criteria': criteria,
                'nogo': nogo})
            
            st.subheader('Meals')
            #st.write(results['meal_output'])
            meals_by_day = results['meal_output'].split("[\\r\\n]+")
            st.write(meals_by_day[0])
            
            st.subheader('Recipes')
            st.write(results['recipes'])
            st.subheader('Shopping List')
            st.write(results['shopping'])

            # check that the ingredients list is actually full 
            # split up meal list into individual meals and generate images

    else:
        st.warning('Please fill in all the fields before generating')


# need to learn how to work with chain outputs much better. The initial prompt works really well, but the rest is a mess.

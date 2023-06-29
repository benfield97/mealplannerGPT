import os 
from apikey import apikey, replicate
import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory
import replicate as rp

os.environ['OPENAI_API_KEY'] = apikey
os.environ['REPLICATE_API_TOKEN'] = replicate


# user interface 
st.title('🍝 Meal Planner 🥗')
st.subheader('Plan meals, get recipes, and make shopping lists with OpenAI')

criteria = st.text_input('Enter some criteria for your meal plan. E.g. Vegetarian, high protein, low carb, etc.')
nogo = st.text_input('Enter any foods you don\'t want to eat. E.g. broccoli, tofu, etc.')
days = st.slider('How many days do you want to plan for?', 1, 7, 3)
meals = st.multiselect('What meals do you want to plan?', ['Breakfast', 'Lunch', 'Dinner'])


# Meal name tempalte
meals_prompt = """You are an expert chef that can cater to any preference.
You are constructing a {days}-day meal plan for a client that includes only the following meals: {meals}.

Plan the meals around the following criteria: 
{criteria}. 

Avoid using the following foods: 
{nogo}.

Provide only the title of each meal.
Provide only a single option for each meal. 
Make sure to provide output in chronological order.

Example:

Day 1: Breakfast: Eggs Benedict
Day 1: Lunch: Chicken Caesar Salad
"""
meals_template = PromptTemplate(template=meals_prompt, input_variables=['days', 'meals', 'criteria', 'nogo'])

# recipe template
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

llm = OpenAI(temperature=0.1, max_tokens=500)

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
            meal_output = mealchain.run(days=days, meals=meals, criteria=criteria, nogo=nogo)

            meal_output = meal_output.splitlines()
            meal_output = [i for i in meal_output if i]
            meal_list = []
            recipe_list = []
            for meal in meal_output:
                meal_list.append(meal.split(': ')[2])
            
            for meal in meal_list:
                recipe = recipechain.run(meal_output=meal)
                recipe_list.append(recipe)

            for i in range(len(meal_output)):
                st.subheader(meal_output[i])

                photo = rp.run("stability-ai/stable-diffusion:db21e45d3f7023abc2a46ee38a23973f6dce16bb082a930b0c49861f96d1e5bf",
                        input={"prompt": f"{meal_list[i]}. Ultra realistic photography, high defintion, recipe photo. HDSLR."})
                st.image(photo)
                
                st.write(recipe_list[i])             

    else:
        st.warning('Please fill in all the fields before generating')



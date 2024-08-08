import os
from enum import Enum
from typing import List, Type, Dict, Any
import pandas as pd
from pydantic import BaseModel, conint, condecimal
from dotenv import load_dotenv
from openai import OpenAI
import instructor
import logging

# Load environment variables
load_dotenv()

# Ensure that the environment variable name matches what is in your .env file
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
client = instructor.from_openai(OpenAI(api_key=OPENAI_API_KEY))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def create_structured_output(text_input: str, system_prompt: str, response_model: Type[BaseModel]) -> BaseModel:
    """Generate a response from a user query."""
    try:
        return client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user",
                 "content": f"Use the given format to extract information from the following input: {text_input}."},
                {"role": "system", "content": system_prompt},
            ],
            response_model=response_model,
        )
    except Exception as e:
        logging.error(f"Error creating structured output: {e}")
        return response_model()


class Range(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class Gender(str, Enum):
    MALE = "MALE"
    FEMALE = "FEMALE"
    NONBINARY = "NONBINARY"


class Relationship(str, Enum):
    SINGLE = "SINGLE"
    IN_A_RELATIONSHIP = "IN_A_RELATIONSHIP"
    MARRIED = "MARRIED"
    DIVORCED = "DIVORCED"
    WIDOWED = "WIDOWED"


class Education(str, Enum):
    NO_HIGH_SCHOOL = "NO HIGH SCHOOL"
    HIGH_SCHOOL = "HIGH SCHOOL"
    SOME_HIGHER_EDUCATION = "SOME HIGHER EDUCATION"
    COLLEGE = "COLLEGE"
    POST_GRADUATE = "POST-GRADUATE"


class Languages(str, Enum):
    ENGLISH = "English"
    SPANISH = "Spanish"
    FRENCH = "French"
    GERMAN = "German"
    ITALIAN = "Italian"
    PORTUGUESE = "Portuguese"
    CHINESE = "Chinese"
    JAPANESE = "Japanese"
    KOREAN = "Korean"
    RUSSIAN = "Russian"
    ARABIC = "Arabic"
    HINDI = "Hindi"
    BENGALI = "Bengali"
    URDU = "Urdu"
    TURKISH = "Turkish"
    VIETNAMESE = "Vietnamese"
    THAI = "Thai"
    SWAHILI = "Swahili"
    POLISH = "Polish"
    DUTCH = "Dutch"


class Fitness(str, Enum):
    ACTIVE = "ACTIVE"
    MODERATE = "MODERATE"
    SEDENTARY = "SEDENTARY"


class UserModel(BaseModel):
    """UserModel for a simple user"""
    age: conint(ge=18, le=85)
    gender: Gender
    pets: bool
    children_under2: bool
    children_3to7: bool
    children_8to12: bool
    children_over12: bool
    relationship_status: Relationship
    city: str = "Berlin"
    neighborhood: str
    home_latitude: str
    home_longitude: str
    education_latitude: str
    education_longitude: str
    work_latitude: str
    work_longitude: str
    gym_latitude: str
    gym_longitude: str
    driver: bool
    income: Range
    education: Education
    preferred_language1: Languages
    preferred_language2: Languages
    solo_activity: bool
    group_activity: bool
    couple_activity: bool
    family_activity: bool
    solo_activity: bool
    works_healthcare_social: bool
    works_it: bool
    works_construction: bool
    works_education: bool
    works_finance: bool
    works_hospitality: bool
    works_service_personalcare: bool
    works_entertainment: bool
    works_retail: bool
    works_transportation: bool
    works_manufacturing: bool
    works_remotely: bool
    impaired_mobility: bool
    impaired_hearing: bool
    impaired_vision: bool
    pregnant: bool
    smoker: bool
    vegan: bool
    vegetarian: bool
    gluten_free: bool
    lactose_free: bool
    nut_free: bool
    organic: bool
    caffeine: bool
    alcohol: bool
    halal: bool
    kosher: bool
    seafood: bool
    italian_food: bool
    greek_food: bool
    turkish_food: bool
    spanish_food: bool
    thai_food: bool
    chinese_food: bool
    japanese_food: bool
    vietnamese_food: bool
    korean_food: bool
    indian_food: bool
    lebanese_food: bool
    ethiopian_food: bool
    moroccan_food: bool
    mexican_food: bool
    brazilian_food: bool
    raw_food: bool
    dessert: bool
    dining_budget: Range
    street_food: bool
    fine_dining: bool
    casual_dining: bool
    relaxing: bool
    lively: bool
    romantic: bool
    adventurous: bool
    educational: bool
    cultural: bool
    luxurious: bool
    nature: bool
    cycling: bool
    swimming: bool
    hiking: bool
    fashion_show: bool
    films_indie: bool
    films_commercial: bool
    museum: bool
    gallery: bool
    theatre: bool
    ballet: bool
    opera: bool
    park: bool
    beach: bool
    party: bool
    concert: bool
    pub: bool
    bar: bool
    club: bool
    cafe: bool
    restaurant: bool
    spa_wellness: bool
    beauty_salon: bool
    craft_workshop: bool
    escape_room: bool
    board_game_club: bool
    sports_play: bool
    sports_watch: bool
    music_play: bool
    music_watch: bool
    theme_park: bool
    artisan_market: bool
    comedy_show: bool
    books: bool
    music_pop: bool
    music_electronic: bool
    music_rock: bool
    music_jazz: bool
    music_classical: bool
    music_indie: bool
    music_hiphop: bool
    music_blues: bool
    music_metal: bool
    music_punk: bool
    music_reggae: bool
    music_folk: bool
    music_soul: bool
    music_alternative: bool
    music_gospel: bool
    music_opera: bool
    music_country: bool
    music_triphop: bool
    music_synthwave: bool
    music_ambient: bool
    shopping_valueseeker: bool
    shopping_aspirational: bool
    shopping_conscious: bool
    shopping_luxury: bool
    shopping_practical: bool
    fitness: Fitness


class LocationRange(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class PopularTime(str, Enum):
    MORNING = "MORNING"
    MIDDAY = "MIDDAY"
    AFTERNOON = "AFTERNOON"
    EVENING = "EVENING"
    NIGHT = "NIGHT"  # TO DO: how to ensure that working times are not incompatible with this?


class TimeSpent(str, Enum):
    SHORT = "SHORT"
    MEDIUM = "MEDIUM"
    LONG = "LONG"


class AvgAge(str, Enum):
    CHILD = "CHILD"
    TEEN = "TEEN"
    YOUNG_ADULT = "YOUNG_ADULT"
    MIDDLE_ADULT = "MIDDLE_ADULT"
    SENIOR_ADULT = "SENIOR_ADULT"


class LocationModel(BaseModel):
    """LocationModel for a location"""
    name: str
    city: str = "Berlin"
    neighborhood: str
    latitude: str
    longitude: str
    car_parking: bool
    bicycle_parking: bool
    accessible_mobility: bool
    accessible_vision: bool
    accessible_hearing: bool
    pregnant_friendly: bool
    smoking: bool
    opening_time_workday: str
    closing_time_workday: str
    opening_time_saturday: str
    closing_time_saturday: str
    opening_time_sunday_holiday: str
    closing_time_sunday_holiday: str
    popular_time_workday: PopularTime
    popular_time_weekend: PopularTime
    dominant_language1: Languages
    dominant_language2: Languages
    time_spent: TimeSpent
    average_age: AvgAge
    adults_only: bool
    pet_friendly: bool
    children_under2: bool
    children_2to7: bool
    children_8to12: bool
    children_over12: bool
    free: bool
    price: LocationRange  # How to make sure that if free = true --> price is always empty?
    solo_friendly: bool
    couple_friendly: bool
    family_friendly: bool
    group_friendly: bool
    remote_work_friendly: bool
    outdoor: bool
    indoor: bool
    active: Fitness
    relaxing: bool
    lively: bool
    romantic: bool
    adventurous: bool
    educational: bool
    cultural: bool
    luxurious: bool
    vegan: bool
    vegetarian: bool
    gluten_free: bool
    lactose_free: bool
    nut_free: bool
    organic: bool
    caffeine_free_options: bool
    caffeine: bool
    alcohol_free_options: bool
    alcohol: bool
    halal: bool
    kosher: bool
    seafood: bool
    italian_food: bool
    greek_food: bool
    turkish_food: bool
    spanish_food: bool
    thai_food: bool
    chinese_food: bool
    japanese_food: bool
    vietnamese_food: bool
    korean_food: bool
    indian_food: bool
    lebanese_food: bool
    ethiopian_food: bool
    moroccan_food: bool
    mexican_food: bool
    brazilian_food: bool
    raw_food: bool
    dessert: bool
    nature: bool
    cycling: bool
    swimming: bool
    hiking: bool
    fashion_show: bool
    films_indie: bool
    films_commercial: bool
    museum: bool
    gallery: bool
    theatre: bool
    ballet: bool
    opera: bool
    park: bool
    beach: bool
    party: bool
    concert: bool
    pub: bool
    bar: bool
    club: bool
    cafe: bool
    restaurant: bool
    street_food: bool
    fine_dining: bool
    casual_dining: bool
    spa_wellness: bool
    beauty_salon: bool
    craft_workshop: bool
    escape_room: bool
    sports_play: bool
    sports_watch: bool
    theme_park: bool
    artisan_market: bool
    comedy_show: bool
    music_pop: bool
    music_electronic: bool
    music_rock: bool
    music_jazz: bool
    music_classical: bool
    music_indie: bool
    music_hiphop: bool
    music_blues: bool
    music_metal: bool
    music_punk: bool
    music_reggae: bool
    music_folk: bool
    music_soul: bool
    music_alternative: bool
    music_gospel: bool
    music_opera: bool
    music_country: bool
    music_triphop: bool
    music_synthwave: bool
    music_ambient: bool
    nearest_station1: str
    nearest_station2: str
    nearest_station3: str


class Motion(str, Enum):
    STATIONARY = "STATIONARY"
    MOVEMENT = "MOVEMENT"


class Group(str, Enum):
    ALONE = "ALONE"
    PAIR = "PAIR"
    GROUP = "GROUP"


class MovementType(str, Enum):
    STATIONARY = "STATIONARY"
    WALKING = "WALKING"
    RUNNING = "RUNNING"
    CYCLING = "CYCLING"
    DRIVING = "DRIVING"
    PUBLIC_TRANSPORT = "PUBLIC_TRANSPORT"


class PlaceType(str, Enum):
    RESIDENTIAL = "RESIDENTIAL"
    WORK = "WORK"
    EDUCATION = "EDUCATION"
    TRANSPORT_STATION = "TRANSPORT_STATION"
    EXERCISE = "EXERCISE"
    DINING = "DINING"
    RETAIL = "RETAIL"
    ENTERTAINMENT = "ENTERTAINMENT"
    HEALTHCARE = "HEALTHCARE"
    CULTURAL = "CULTURAL"
    SOCIAL = "SOCIAL"
    PERSONAL_SERVICES = "PERSONAL_SERVICES"
    GOVERNMENT = "GOVERNMENT"


class UserActivityModel(BaseModel):
    user_id: conint(ge=1)
    event_id: conint(ge=1)
    start_timestamp: str  # Timestamp in a format like "YYYY-MM-DD HH:MM:SS"
    duration_minutes: conint(ge=1)
    start_latitude: str
    start_longitude: str
    end_latitude: str
    end_longitude: str
    movement: List[MovementType]
    private: bool
    indoor: bool
    group_size: Group
    place_type: List[PlaceType]
    name: str
    payment_amount: condecimal(max_digits=4, decimal_places=2, ge=0, le=70)


def save_to_versioned_csv(data: pd.DataFrame, base_filename: str):
    """Save DataFrame to a versioned CSV file."""
    version = 1
    filename = f"{base_filename}_v{version}.csv"
    while os.path.exists(filename):
        version += 1
        filename = f"{base_filename}_v{version}.csv"
    data.to_csv(filename, index=False)
    logging.info(f"Data saved to {filename}")


def generate_user_data(prompt: str, system_prompt: str, setup: List[List[Any]]) -> List[Dict[str, Any]]:
    """Generate user data and return as a list of dictionaries."""
    results = []
    user_id = 1  # Initialize user_id
    for item in setup:
        result = create_structured_output(text_input=item[0], system_prompt=item[1], response_model=item[2])
        result_data = result.dict()
        result_data['user_id'] = user_id
        result_data['prompt'] = item[0]  # Include the prompt used for the user
        results.append(result_data)
        user_id += 1
    return results


def generate_location_data(prompt: str, system_prompt: str, setup: List[List[Any]]) -> List[Dict[str, Any]]:
    """Generate location data and return as a list of dictionaries."""
    results = []
    location_id = 1  # Initialize location_id
    for item in setup:
        result = create_structured_output(text_input=item[0], system_prompt=item[1], response_model=item[2])
        result_data = result.dict()
        result_data['location_id'] = location_id
        results.append(result_data)
        location_id += 1
    return results


def generate_activity_data(users: List[Dict[str, Any]], system_prompt: str, setup: List[List[Any]],
                           events_per_user: int) -> List[Dict[str, Any]]:
    """Generate activity data and return as a list of dictionaries, ensuring no temporal overlap."""
    results = []
    event_id = 1  # Initialize event_id

    for user in users:
        user_id = user['user_id']
        user_prompt = user['prompt']

        # To keep track of the last end time of activities for this user
        last_end_time = None

        for _ in range(events_per_user):
            for item in setup:
                # Customize the prompt based on the user_id and user_prompt
                prompt = item[0].replace("{user_prompt}", user_prompt).replace("{user_id}", str(user_id))
                result = create_structured_output(text_input=prompt, system_prompt=item[1], response_model=item[2])
                result_data = result.dict()

                # Convert timestamps to datetime objects for comparison
                start_time = pd.to_datetime(result_data['start_timestamp'])

                # Ensure the start_time is timezone-naive
                if start_time.tzinfo is not None:
                    start_time = start_time.tz_localize(None)

                # Ensure that the start time is after the last end time
                if last_end_time is not None:
                    if start_time <= last_end_time:
                        # Adjust start and end times to avoid overlap
                        start_time = last_end_time + pd.Timedelta(minutes=1)
                        result_data['start_timestamp'] = start_time.strftime("%Y-%m-%d %H:%M:%S")

                        result_data['end_timestamp'] = (
                                    start_time + pd.Timedelta(minutes=result_data['duration_minutes'])).strftime(
                            "%Y-%m-%d %H:%M:%S")

                    else:
                        # Compute end time based on the start time and duration

                        result_data['end_timestamp'] = (
                                    start_time + pd.Timedelta(minutes=result_data['duration_minutes'])).strftime(
                            "%Y-%m-%d %H:%M:%S")

                else:
                    # Compute end time based on the start time and duration

                    result_data['end_timestamp'] = (
                                start_time + pd.Timedelta(minutes=result_data['duration_minutes'])).strftime(
                        "%Y-%m-%d %H:%M:%S")

                # Update the last end time
                last_end_time = pd.to_datetime(result_data['end_timestamp'])

                # Ensure the last_end_time is timezone-naive
                if last_end_time.tzinfo is not None:
                    last_end_time = last_end_time.tz_localize(None)

                # Ensure only necessary fields are included in the final result
                final_result_data = {
                    'user_id': user_id,
                    'event_id': event_id,
                    'start_timestamp': result_data['start_timestamp'],
                    'duration_minutes': result_data['duration_minutes'],
                    'start_latitude': result_data['start_latitude'],
                    'start_longitude': result_data['start_longitude'],
                    'end_latitude': result_data['end_latitude'],
                    'end_longitude': result_data['end_longitude'],
                    'movement': result_data['movement'],
                    'private': result_data['private'],
                    'indoor': result_data['indoor'],
                    'group_size': result_data['group_size'],
                    'place_type': result_data['place_type'],
                    'name': result_data['name'],
                    'payment_amount': result_data['payment_amount']
                }

                results.append(final_result_data)
                event_id += 1

    return results


def main():
    prompt_user = "GIVEN all these information and your knowledge about people in general, generate a detailed and realistic user profile."
    prompt_location = "GIVEN all these information and your knowledge about places in general, generate a comprehensive description of an actual, well-known place in Berlin (e.g., Treptower Park or Museum of Natural History) considering diverse attributes such as types of activities, food options, music genres, and accessibility."
    prompt_activity = "Generate an activity log for user with ID {user_id} and profile: {user_prompt}."

    system_prompt_user = (
        "You are a creative expert in generating mock datasets containing information about users living in Berlin (demographics, hobbies, music/food preferences...) "
        "based on just a few facts you know about the certain user. You confidently use your knowledge about people to extrapolate about other user characteristics. "
        "The datasets you come up with will be used to map out user characteristics for a travel recommendation app. "
        "Generate a realistic and internally consistent user profile. When unsure which value to assign to a field, choose the most probable value based on what you know (about the user and about people in general, what traits go together). "
        "For example, if the user is described as a minimum-wage worker with 3 children interested in sports, fill in appropriate details such as 'income: LOW' and 'fitness: ACTIVE', likewise with all the other traits, even if they don't seem directly related. "
        "Keep in mind that every user has a home location, but not every user has all of these: a work locaton, an education location and a gym location."
        "Conclude on your own whether they work, go to school and/or to the gym, and generate the data accordingly; "
        "feel free to keep the coordinates for the corresponding place (woork/school/gym) empty. "
    )
    system_prompt_location = (
        "You are a creative expert in generating mock datasets containing information about places in Berlin (accessibility, atmosphere, activities...) "
        "based on just a few facts you know about a certain location/venue. You confidently use your knowledge about tourist attractions and venues to extrapolate what other characteristics the place has. "
        "The datasets you come up with will be used to map out location characteristics for a travel recommendation app. "
        "When generating locations, always use real place names from Berlin. For example, if describing a popular nightclub, use actual names like 'Berghain'. "
        "If it's an outdoor park, use names like 'Tempelhofer Feld' or 'Volkspark Friedrichshain'. Ensure all other information fits logically with these real locations."
    )
    system_prompt_activity = (
        "You are a creative expert in generating mock datasets containing information about user activities in Berlin. "
        "Based on the user profile provided, generate a realistic and internally consistent activity log for this user. "
        "Have the activities (including their start and end locations) be distributed across very different "
        "and widely geographically distributed, but real locations in Berlin, including shops, museums, parks..."
        "Don't assume that users spend most of their time in residential areas, include more activities in public places."
        "Make sure that the activities are logical, for example moving (walking, driving...) means arriving at a "
        "different location than where you started from. "
        "Similarly, residential places are always private and never public, so they don't have a name associated with them."
    )

    user_generation_setup = [
        ["He is employed, working-class, with 3 children and is interested in sports", system_prompt_user, UserModel],
        ["They are in college, come from a rich family and have expensive taste", system_prompt_user, UserModel],
        ["She is retired and likes to spend time with her cats and grandchildren, likes the arts", system_prompt_user,
         UserModel]

    ]
    location_generation_setup = [
        ["A park with food stands and places for playing sports", system_prompt_location, LocationModel],
        ["A popular nightclub for university students", system_prompt_location, LocationModel],
        ["An alternative bar you can also eat in", system_prompt_location, LocationModel]
    ]
    activity_generation_setup = [
        [prompt_activity, system_prompt_activity, UserActivityModel]
    ]

    logging.info("Generating user data...")
    user_results = generate_user_data(prompt_user, system_prompt_user, user_generation_setup)
    logging.info("User data generation complete.")

    logging.info("Generating location data...")
    location_results = generate_location_data(prompt_location, system_prompt_location, location_generation_setup)
    logging.info("Location data generation complete.")

    # Define the number of events per user
    events_per_user = 5

    logging.info("Generating activity data...")
    activity_results = generate_activity_data(user_results, system_prompt_activity, activity_generation_setup,
                                              events_per_user)

    logging.info("Activity data generation complete.")

    # Remove 'prompt' field from user_results before saving
    for user in user_results:
        user.pop('prompt', None)

    # Convert results to DataFrames
    user_df = pd.DataFrame(user_results)
    location_df = pd.DataFrame(location_results)
    activity_df = pd.DataFrame(activity_results)

    # Reorder columns in user_df to make 'user_id' the first column
    user_columns = ['user_id'] + [col for col in user_df.columns if col != 'user_id']
    user_df = user_df[user_columns]

    # Reorder columns in location_df to make 'location_id' the first column
    location_columns = ['location_id'] + [col for col in location_df.columns if col != 'location_id']
    location_df = location_df[location_columns]

    # Save to versioned CSV
    save_to_versioned_csv(user_df, "user_data")
    save_to_versioned_csv(location_df, "location_data")
    save_to_versioned_csv(activity_df, "activity_data")


if __name__ == "__main__":
    main()

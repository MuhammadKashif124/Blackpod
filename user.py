import os
import logging
from dotenv import load_dotenv
import hashlib
from pymongo import MongoClient
import certifi
import gradio as gr

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('user_manager.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

class UserManager:
    def __init__(self):
        # Validate environment variables
        mongodb_uri = os.getenv('MONGODB_URI')
        database_name = os.getenv('DATABASE_NAME')
        collection_name = os.getenv('COLLECTION_NAME')
        
        if not all([mongodb_uri, database_name, collection_name]):
            logger.error("Missing required environment variables")
            raise ValueError("Missing required environment variables: MONGODB_URI, DATABASE_NAME, COLLECTION_NAME")
        
        try:
            # Simplified connection for local MongoDB
            self.client = MongoClient(mongodb_uri)
            self.db = self.client[database_name]
            self.collection = self.db[collection_name]
            logger.info("Successfully connected to MongoDB")
        except Exception as e:
            logger.error(f"Error connecting to MongoDB: {str(e)}")
            raise

    def create_user(self, username, password, org_key):
        if not all([username, password, org_key]):
            return "All fields are required!"
        if self.collection.find_one({"username": username, "org_key": org_key}):
            return "User already exists in this organization!"
        hashed_password = self.hash_password(password)
        self.collection.insert_one({"username": username, "password": hashed_password, "org_key": org_key})
        return f"User {username} created successfully!"

    def login(self, username, password, org_key):
        if not all([username, password, org_key]):
            return "All fields are required!"
        user = self.collection.find_one({"username": username, "org_key": org_key})
        if not user:
            return "User not found or invalid organization key!"
        if user['password'] == self.hash_password(password):
            return f"Login successful! Organization Key: {user['org_key']}"
        else:
            return "Invalid password!"

    def hash_password(self, password):
        return hashlib.sha256(password.encode()).hexdigest()

    def get_users(self, page=1, per_page=5, org_key=None):
        if page < 1:
            page = 1
        query = {}
        if org_key:
            query["org_key"] = org_key
        users = list(self.collection.find(query).skip((page - 1) * per_page).limit(per_page))
        return [(user["username"], user["org_key"]) for user in users], self.collection.count_documents(query)

    def update_user(self, username, org_key, new_password, new_org_key):
        if not all([username, org_key, new_password, new_org_key]):
            return "All fields are required!"
        user = self.collection.find_one({"username": username, "org_key": org_key})
        if not user:
            return "User not found or invalid organization key!"
        hashed_password = self.hash_password(new_password)
        self.collection.update_one({"username": username, "org_key": org_key}, {"$set": {"password": hashed_password, "org_key": new_org_key}})
        return f"User {username} updated successfully!"

    def delete_user(self, username, org_key):
        if not all([username, org_key]):
            return "All fields are required!"
        result = self.collection.delete_one({"username": username, "org_key": org_key})
        if result.deleted_count == 0:
            return "User not found or invalid organization key!"
        return f"User {username} deleted successfully!"

# Gradio UI
def create_user_ui(username, password, org_key):
    user_manager = UserManager()
    return user_manager.create_user(username, password, org_key)

def login_ui(username, password, org_key):
    user_manager = UserManager()
    return user_manager.login(username, password, org_key)

def view_users_ui(page, org_key):
    user_manager = UserManager()
    users, total_users = user_manager.get_users(page=page, org_key=org_key)
    return users, total_users

def update_user_ui(username, org_key, new_password, new_org_key):
    user_manager = UserManager()
    return user_manager.update_user(username, org_key, new_password, new_org_key)

def delete_user_ui(username, org_key):
    user_manager = UserManager()
    return user_manager.delete_user(username, org_key)

# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("## User Management System")
    
    with gr.Tab("Create User"):
        username_input = gr.Textbox(label="Username")
        password_input = gr.Textbox(label="Password", type="password")
        org_key_input = gr.Textbox(label="Organization Key")
        create_button = gr.Button("Create User")
        create_output = gr.Textbox(label="Output", interactive=False)
        create_button.click(create_user_ui, inputs=[username_input, password_input, org_key_input], outputs=create_output)

    with gr.Tab("Login"):
        login_username_input = gr.Textbox(label="Username")
        login_password_input = gr.Textbox(label="Password", type="password")
        login_org_key_input = gr.Textbox(label="Organization Key")
        login_button = gr.Button("Login")
        login_output = gr.Textbox(label="Output", interactive=False)
        login_button.click(login_ui, inputs=[login_username_input, login_password_input, login_org_key_input], outputs=login_output)

    with gr.Tab("View Users"):
        org_key_filter = gr.Dropdown(label="Filter by Organization Key", choices=[], value=None)
        page_input = gr.Number(label="Page", value=1, precision=0)
        view_button = gr.Button("View Users")
        users_output = gr.Dataframe(label="Users", headers=["Username", "Organization Key"])
        total_users_output = gr.Textbox(label="Total Users", interactive=False)

        def update_org_keys():
            user_manager = UserManager()
            org_keys = list(user_manager.collection.distinct("org_key"))
            return org_keys

        # Load organization keys when the tab is opened
        org_key_filter.choices = update_org_keys()

        def update_users(page, org_key):
            users, total_users = view_users_ui(page, org_key)
            total_users_output.value = str(total_users)
            return users

        view_button.click(update_users, inputs=[page_input, org_key_filter], outputs=users_output)

        # Update User Section
        update_username_input = gr.Textbox(label="Username to Update")
        update_org_key_input = gr.Textbox(label="Organization Key")
        new_password_input = gr.Textbox(label="New Password", type="password")
        new_org_key_input = gr.Textbox(label="New Organization Key")
        update_button = gr.Button("Update User")
        update_output = gr.Textbox(label="Update Output", interactive=False)
        update_button.click(update_user_ui, inputs=[update_username_input, update_org_key_input, new_password_input, new_org_key_input], outputs=update_output)

        # Delete User Section
        delete_username_input = gr.Textbox(label="Username to Delete")
        delete_org_key_input = gr.Textbox(label="Organization Key")
        delete_button = gr.Button("Delete User")
        delete_output = gr.Textbox(label="Delete Output", interactive=False)
        delete_button.click(delete_user_ui, inputs=[delete_username_input, delete_org_key_input], outputs=delete_output)

# Launch the Gradio app
if __name__ == "__main__":
    demo.launch()

from database import init_db, add_user

# 1. Initialize the tables (creates 'users', 'logs', etc.)
print("Initializing tables...")
init_db()

# 2. Add your first Admin user
# Replace 'admin_pass' with a strong password you want to use
username = "revanth_admin"
password = "admin_pass" 
name = "Revanth"

print(f"Creating admin user: {username}...")
success = add_user(username, password, name, roles=['admin', 'user'])

if success:
    print("✅ Admin user created successfully!")
else:
    print("❌ User already exists or error occurred.")
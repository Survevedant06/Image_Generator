"""Seed script to create demo user accounts with password: password123 and partition products by field."""
import auth
import db

def seed():
    db.init_db()
    PASSWORD = "password123"
    hashed = auth.hash_password(PASSWORD)

    USERS = [
        ("alex_studio",    "alex@studio.com",   "Alex Studio (Fashion & Lifestyle)"),
        ("morgan_catalog", "morgan@catalog.com", "Morgan Catalog (Tech & Electronics)"),
        ("user1",          "user1@studio.com",  "User 1 (Fashion & Lifestyle)"),
        ("user2",          "user2@tech.com",    "User 2 (Tech & Electronics)"),
    ]

    for username, email, full_name in USERS:
        existing = db.get_user_by_username(username)
        if existing:
            conn = db.get_db_connection()
            conn.execute("UPDATE users SET password_hash = ? WHERE username = ?", (hashed, username))
            conn.commit()
            conn.close()
            print(f"[UPDATED] {username} password reset to: {PASSWORD}")
        else:
            user = db.create_user(username, email, hashed, full_name)
            print(f"[CREATED] {username} (id={user['id']})")

    alex = db.get_user_by_username("alex_studio")
    morgan = db.get_user_by_username("morgan_catalog")
    if alex and morgan:
        conn = db.get_db_connection()
        conn.execute("UPDATE products SET user_id = ? WHERE category_id IN (25, 30, 28, 29, 33)", (alex["id"],))
        conn.execute("UPDATE products SET user_id = ? WHERE category_id IN (26, 18, 31, 32, 27)", (morgan["id"],))
        conn.commit()
        c_alex = conn.execute("SELECT COUNT(*) FROM products WHERE user_id = ?", (alex["id"],)).fetchone()[0]
        c_morgan = conn.execute("SELECT COUNT(*) FROM products WHERE user_id = ?", (morgan["id"],)).fetchone()[0]
        conn.close()
        print(f"alex_studio catalog: {c_alex} products")
        print(f"morgan_catalog catalog: {c_morgan} products")

    print("\nAll done! All accounts share the password: password123")

if __name__ == "__main__":
    seed()

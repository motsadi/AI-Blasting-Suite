import streamlit_authenticator as stauth
print("Enter a password to hash:")
try:
    pwd = input().strip()
except EOFError:
    pwd = ""
if not pwd:
    print("No password entered."); exit(1)
hashes = stauth.Hasher([pwd]).generate()
print("\nHashed password:\n", hashes[0])
print("\nCopy into secrets under credentials.usernames.<username>.password")

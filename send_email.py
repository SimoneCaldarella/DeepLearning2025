import urllib.parse
import pandas as pd

# Function to prepare the email body and generate a mailto link
def prepare_mailto_link(df):
    mailto_links = []
    
    for row in df.values.tolist():
        members = []
        emails = []
        # Extracting the columns: emails, names, user-group, password
        main_email = row[1]

        if pd.isna(main_email):
            print("Main email is missing, skipping this row.")
            continue

        name1 = f"{row[2]} {row[3]}" if not pd.isna(row[2]) else None
        try: id1 = int(row[4])
        except: id1 = None
        email1 = row[5] if not pd.isna(row[5]) else None

        name2 = f"{row[6]} {row[7]}" if not pd.isna(row[6]) else None
        try: id2 = int(row[8])
        except: id2 = None
        email2 = row[9] if not pd.isna(row[9]) else None

        name3 = f"{row[10]} {row[11]}" if not pd.isna(row[10]) else None
        try: id3 = int(row[12])
        except: id3 = None
        email3 = row[13] if not pd.isna(row[13]) else None

        user_group = row[14]
        password = row[15]
        login_url = row[16]

        emails.append(main_email)

        if name1: 
            members.append(f"- {name1} (ID: {id1}) | {email1}")
            if email1 and not email1 == main_email: emails.append(email1)
        if name2: 
            members.append(f"- {name2} (ID: {id2}) | {email2}")
            if email2 and not email2 == main_email: emails.append(email2)
        if name3: 
            members.append(f"- {name3} (ID: {id3}) | {email3}")
            if email3 and not email3 == main_email: emails.append(email3)
        
        
        members_body = '\n'.join(members)
        
        # Email body (URL-encoded)
        body = f"""Dear participants of the DeepLearning 2025 course,

We are pleased to inform you that you have been granted access to the AWS Instances for the DeepLearning 2025 course.

The following members are part of your group:
{str(members_body)}

This is your account information for AWS Instances:
Username: {user_group}
Password: {password}

Access the platform using the following link:
{login_url}

Best regards,
Thomas De Min, 
Simone Caldarella
"""
            # URL-encode the subject and body
        subject = "AWS Instances for DeepLearning 2025 course"
        encoded_subject = urllib.parse.quote(subject)
        encoded_body = urllib.parse.quote(body)
        cc_emails = "cc="+','.join(emails[1:]) if len(emails) >= 2 else ""

        # Create the mailto link with subject, body, and recipients
        mailto_link = f"mailto:{emails[0]}?{cc_emails}&subject={encoded_subject}&body={encoded_body}"
        mailto_links.append(mailto_link)

    return mailto_links

# Usage
if __name__ == "__main__":
    import argparse

    argparser = argparse.ArgumentParser(description="Generate mailto links for AWS Instances access.")
    argparser.add_argument("-f", "--file", type=str, help="Path to the CSV file containing group information.", required=True)
    args = argparser.parse_args()
    file_path = args.file
    df = pd.read_csv(file_path, header=None)
    print(df)
    mailto_links = prepare_mailto_link(df)

    # Print all generated mailto links
    print("Generated mailto links:")
    print("\n\n".join(mailto_links))

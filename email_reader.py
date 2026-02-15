from imapclient import IMAPClient
import pyzmail
import pandas as pd
import os

EMAIL = "deepthi.ai27@gmail.com"
APP_PASSWORD = "hzwdpmrqzseuvete"
SAVE_DIR = "data"

os.makedirs(SAVE_DIR, exist_ok=True)

with IMAPClient("imap.gmail.com") as server:
    server.login(EMAIL, APP_PASSWORD)
    server.select_folder("INBOX")

    messages = server.search(['UNSEEN'])

    for uid in messages:
        raw = server.fetch(uid, ['BODY[]', 'FLAGS'])
        message = pyzmail.PyzMessage.factory(raw[uid][b'BODY[]'])

        if message.mailparts:
            for part in message.mailparts:
                if part.filename and part.filename.endswith(".csv"):
                    filepath = os.path.join(SAVE_DIR, part.filename)
                    with open(filepath, "wb") as f:
                        f.write(part.get_payload())
                    print("Downloaded:", part.filename)

# tgBotToObsidian
This software was forked from [tg2obsidian](https://github.com/dimonier/tg2obsidian) and updated to work with modern AIOgram version.

## About
The bot is designed to store messages sent to it locally on disk.
The main purpose is to form notes from messages in Obsidian.
Texts are reformatted for Markdown, which can be used not only in Obsidian.
The bot understands both messages sent directly to it and forward ones.


## Features
- All messages are grouped by date � one note per day � or stored in a single note.
- Each message in a note has a header with a date and time stamp.
- Formatting of messages and captions is preserved or ignored depending on settings.
- For forwarded messages, information about the origin is added.
- Photos, animations, videos, and documents are saved to the vault and embedded in the note.
- For contacts, YAML front matter and vcard are saved.
- For locations, relevant links to Google and Yandex maps are created.
- It is possible to convert notes with certain keywords into a task.
- It is possible to tag notes with certain keywords.
- It is possible to recognize speech from voice notes and audio messages. In this case, the Bot sends the recognized text as a response to the original message.

## Setup
1. Install [Python 3.10+](https://python.org/)
2. Install script dependencies:
```
pip install aiogram
pip install beautifulsoup4
pip install lxml
```

3. Install [Whisper](https://github.com/openai/whisper) and Pytorch modules if you need voice messages get recognized to text:

```
pip install -U openai-whisper
pip install torch
```

4. Install compiled [FFMPEG](https://ffmpeg.org/download.html) and add the path to the executable (in Windows � ffmpeg.exe) to the ==path== environment variable. Go to the folder containing this script and make sure that ==ffmpeg.exe== could be started there.
5. Create your own bot using https://t.me/BotFather.
6. Rename config_sample.py to config.py.
7. Paste the token received from ==@botfather== into the appropriate parameter in ==config.py== and change the rest of the parameters in ==config.py== as desired.
8. (Optional) Add the bot created above to a private Telegram group and make it administrator so it can read messages.
 

## Usage

1. Send or forward messages that should go to your Obsidian vault to the private Telegram group or directly to your Telegram bot.
2. Run Bot:
```
python tg2obsidian_bot.py
```

- You can keep Bot running indefinitely on a computer or server that is permanently turned on. In this case, it will recognize speech and create/update notes in Obsidian in real time.
- If you only turn your computer on when you're using it, run Bot directly when you need to get Obsidian messages, and close the program when you've received all the messages.

**Important!** Bot can only retrieve messages for the last 24 hours. Messages sent earlier and not retrieved in a timely manner will not be received by Bot and saved in the vault.
## Known issues

Check in the [Issues](https://github.com/chieftain-yu/tgBotToObsidian/issues) section.
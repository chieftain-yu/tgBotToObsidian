import gettext
import token
import aiohttp
import asyncio
#import config_sample as config
import config
#You should put your real token to config and use it in your real config.py
import logging
import os
import re
#import sys
import time

#from os import getenv
from bs4 import BeautifulSoup
from datetime import datetime as dt
from pathlib import Path
import urllib.request

from aiogram import Bot, Dispatcher, F, Router, types, html
#from aiogram.enums import ParseMode
from aiogram.filters import Command, CommandStart
from aiogram.types import ContentType, File, Message, MessageEntity
from aiogram.utils.markdown import hbold

#Заявляем класс заметки
class Note:
    def __init__(self,
                 text = "",
                 date = dt.now().strftime('%Y-%m-%d'),
                 time = dt.now().strftime('%H:%M:%S')):
        self.text = text
        self.date = date
        self.time = time
        
basic_log = False
debug_log = False
#Переработка пути - неверно работало в Fedora40
def osPathJoin (path: str, file:str) -> str:
    return os.path.normpath (path+os.sep+file)

def new_func():
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO, filename = 'bot.log', encoding = 'UTF-8', datefmt = '%Y-%m-%d %H:%M:%S')

if 'log_level' in dir(config) and config.log_level >= 1:
    basic_log = True
    if config.log_level >= 2:
        debug_log = True
    new_func()
    log = logging.getLogger()

# Functions

# Определяем функцию простейшего логирования
def log_basic(text: str):
    if basic_log:
        log.info(text)
    
# Определяем функцию дебаг-логирования
def log_debug(text: str):
    if debug_log:
        log.info(text)

# Определяем логирование сообщения в зависимости от уровня логирования
def log_message(message):
    # Saving of the whole message into the incoming message log just in case
    if debug_log:
        curr_date = dt.now().strftime('%Y-%m-%d')
        curr_time = dt.now().strftime('%H:%M:%S')
        file_name = 'messages-' + curr_date + '.txt'
        with open(file_name, 'a', encoding='UTF-8') as f:
            print(curr_time + '  ', list(message), '\n', file = f)
        log_debug(f'Message content saved to {file_name}')
        
# Определяем функцию сохранения сообщения в Obsidian
def save_message(note: Note) -> None:
    curr_date = note.date
    curr_time = note.time
    if one_line_note():
        # Replace all line breaks with spaces and make simple time stamp
        note_body = note.text.replace('\n', ' ')
        note_text = check_if_task(check_if_negative(f'[[{curr_date}]] - {note_body}\n'))
    else:
        # Keep line breaks and add a header with a time stamp
        note_body = check_if_task(check_if_negative(note.text))
        note_text = f'#### [[{curr_date}]] {curr_time}\n{note_body}\n\n'
    with open(get_note_name(curr_date), 'a', encoding='UTF-8') as f:
        f.write(note_text)

if 'log_level' in dir(config) and config.log_level >= 1:
    basic_log = True
    if config.log_level >= 2:
        debug_log = True
    logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO, filename = 'bot.log', encoding = 'UTF-8', datefmt = '%Y-%m-%d %H:%M:%S')
    log = logging.getLogger()

# Bot token can be obtained via https://t.me/BotFather
token=os.environ.get("tgBot2ObsToken")    
print (f'{token}')
bot = Bot(token)

# Готовим распознавание речи
if config.recognize_voice:
    import torch
    import gc
    print('Prepared for speech-to-text recognition')

# Функция записи текста в заметку
def note_from_message(message: Message):
    msg_date = message.date.strftime('%Y-%m-%d')
    msg_time = message.date.strftime('%H:%M:%S')
    note = Note(date=msg_date, time=msg_time)
    return note

# Проверка необходимости сохранения форматирования
def format_messages() -> bool:
    format_messages = True if 'format_messages' not in dir(config) or config.format_messages else False
    return format_messages

# Проверка необходимости указания ссылок
def create_link_info() -> bool:
    return False if 'create_link_info' not in dir(config) else config.create_link_info

# Проверка  на то, не задача ли это
def check_if_task(note_body) -> str:
    is_task = False
    for keyword in config.task_keywords:
        if keyword.lower() in note_body.lower(): is_task = True
    if is_task: note_body = '- [ ] ' + note_body
    return note_body

# Автотегирование по негативным словам (см. config.py)
def check_if_negative(note_body) -> str:
    is_negative = False
    for keyword in config.negative_keywords:
        if keyword.lower() in note_body.lower(): is_negative = True
    if is_negative: note_body += f'\n{config.negative_tag}'
    return note_body

# Поиск первого непробельного символа
def content_index(c: str) -> int:
    ret = 0
    for i in c:
       if not i.isspace():
           return ret
       ret += 1
    return -1

# Поиск форматирований
def find_entities(message: Message) -> str:
    
    return 0

# Функция переформатирования текста для md
async def embed_formatting(message: Message) -> str:
    # If the message contains any formatting (inclusing inline links), add corresponding Markdown markup
    #todo: find why the message is not subscriptable
    note = message.text

    if not format_messages():
        return note

    if not message.entities:
        return note

    entities = message.entities
    formatted_note = ''
    try:
        note_u16 = to_u16(note)
        formatted_note = parse_entities(note_u16, entities, 0, len(note_u16))
        if create_link_info() and is_single_url(message):
            url_entity = entities[0]
            url = url_entity.get_text(note) if url_entity['type'] == "url" else url_entity['url']
            formatted_note += await get_url_info_formatting(url)
    except Exception as e:
        # If the message does not contain any formatting
        await message.reply(f'🤷‍♂️ {e}')
        print (f'embed_formatting exception!')
        formatted_note = note
    return formatted_note

# Функция получения информации для форварднутого сообщения
def get_forward_info(m: Message) -> str:
    # If the message is forwarded, extract forward info and make up forward header
    forward_info = ''
    post = 'message'
    user = ''
    chat = ''
    forwarded = False
    if m.forward_from_chat:
        forwarded = True
        # Todo: unversal parser of chat id. Currently works for sure for channels only
        chat_id = str(m.forward_from_chat.id)[4:]
        if m.forward_from_chat.username:
            chat_name = f'[{m.forward_from_chat.title}](https://t.me/{m.forward_from_chat.username})'
        else:
            chat_name = f'{m.forward_from_chat.title}'
        chat = f'from {m.forward_from_chat.type} {chat_name}'

        if m.forward_from_message_id:
            msg_id = str(m.forward_from_message_id)
            post = f'[message](https://t.me/c/{chat_id}/{msg_id})'
            
    if m.forward_from:
        forwarded = True
        real_name = ''
        if m.forward_from.first_name: 
            real_name += m.forward_from.first_name
        if m.forward_from.last_name: 
            real_name += ' ' + m.forward_from.last_name
        real_name = real_name.strip()
        if m.forward_from.username:
            user = f'by [{real_name}](https://t.me/{m.forward_from.username})'
        else:
            user = f'by {real_name}'
    elif m.forward_sender_name:
        forwarded = True
        user = f'by {m.forward_sender_name}'

    forward_info = ' '.join([item for item in [post, chat, user] if len(item) > 0])

    if forwarded:
        result = bold(f'Forwarded {forward_info}') + '\n'
    else:
        result = ''

    return result

async def handle_file(file: File, file_name: str, path: str):
    Path(f"{path}").mkdir(parents=True, exist_ok=True)
    await bot.download_file(file_path=file.file_path, destination=f"{path}/{file_name}")

async def get_formatted_caption(message: Message) -> str:

    if message.caption:
        doc_message = {
            'text': message.caption,
            'entities': message.caption_entities,
            }
        return await embed_formatting(doc_message)
    else:
        return ''
    
def get_note_file_name_parts(curr_date):
    filename_part1 = config.note_prefix if 'note_prefix' in dir(config) else ''
    filename_part3 = config.note_postfix if 'note_postfix' in dir(config) else ''
    filename_part2 = curr_date if 'note_date' in dir(config) and config.note_date is True else ''
    return [filename_part1, filename_part2, filename_part3]

def get_note_name(curr_date) -> str:
    parts = get_note_file_name_parts(curr_date)
    return osPathJoin(config.inbox_path, ''.join(parts) + '.md')

def create_media_file_name(message: Message, suffix = 'media', ext = 'jpg') -> str:
    # ToDo: переделать на дату отправки сообщения
#todo получать локальное время вместо UTC
    curr_date = get_curr_date()
    parts = get_note_file_name_parts(curr_date)
    # ToDo: добавить в имя файлаusername исходного канала или пользователя
    # Если присутствует forward_from - оттуда, иначе из from

    # Строим среднюю часть имени без лишних - и _
    note_name = re.sub("[-_]+", "-", f'{parts[0]}{parts[2]}'.strip('-_'))

    return f'{curr_date}_{note_name}_{suffix}.{ext}'

def get_curr_date() -> str:
    return dt.now().strftime('%Y-%m-%d')

def one_line_note() -> bool:
    one_line_note = False if 'one_line_note' not in dir(config) or config.one_line_note == False else True
    return one_line_note

# All handlers should be attached to the Router (or Dispatcher)
dp = Dispatcher()

#returns (ws?, content?, ws?)
def partition_string(text: str) -> tuple:
    start = content_index(text)
    if start == -1:
        return (text,'','')
    end = content_index(text[::-1])
    end = len(text) if end == -1 else len(text) - end
    return (text[:start], text[start:end], text[end:])

def to_u16(text: str) -> bytes:
    return text.encode('utf-16-le')

def from_u16(text: bytes) -> str:
    return text.decode('utf-16-le')

formats = {'bold': ('**', '**'),
           'italic': ('_', '_'),
           'underline': ('<u>', '</u>'),
           'strikethrough': ('~~', '~~'),
           'code': ('`', '`'),
}

#!!!
def parse_entities(text: bytes,
    entities: list[MessageEntity],
    offset: int,
    end: int) -> str:
    formatted_note = ''

    for entity_index, entity in enumerate(entities):
        entity_start = entity.offset * 2
        if entity_start < offset:
            continue
        if entity_start > offset:
            formatted_note += from_u16(text[offset:entity_start])
        offset = entity_end = entity_start + entity.length * 2
        format = entity.type
        if format == 'pre':
            pre_content = from_u16(text[entity_start:entity_end])
            content_parts = partition_string(pre_content)
            formatted_note += '```'
            if (len(content_parts[0]) == 0 and
                content_parts[1].find('\n') == -1):
                formatted_note += '\n'
            formatted_note += pre_content
            if content_parts[2].find('\n') == -1:
                formatted_note += '\n'
            formatted_note += '```'
            if (len(text) - entity_end < 2 or
               from_u16(text[entity_end:entity_end+2])[0] != '\n'):
                formatted_note += '\n'
            continue
        # parse nested entities for exampe: "**bold _italic_**
        sub_entities = [e for e in entities[entity_index + 1:] if e.offset * 2 < entity_end]
        parsed_entity = parse_entities(text, sub_entities, entity_start, entity_end)
        content_parts = partition_string(parsed_entity)
        content = content_parts[1]
        if format in formats:
            format_code = formats[format]
            formatted_note += content_parts[0]
            i = 0
            while i < len(content):
                index = content.find('\n\n', i) # inline formatting acros paragraphs, need to split
                if index == -1:
                    formatted_note += format_code[0] + content[i:] + format_code[1]
                    break
                formatted_note += format_code[0] + content[i:index] + format_code[1]
                i = index
                while i < len(content) and content[i] == '\n':
                    formatted_note += '\n'
                    i += 1
            formatted_note += content_parts[2]
            continue
        if format == 'mention':
            formatted_note += f'{content_parts[0]}[{content}](https://t.me/{content[1:]}){content_parts[2]}'
            continue
        if format == 'text_link':
            formatted_note += f'{content_parts[0]}[{content}]({entity["url"]}){content_parts[2]}'
            continue
        # Not processed (makes no sense): url, hashtag, cashtag, bot_command, email, phone_number
        # Not processed (hard to visualize using Markdown): spoiler, text_mention, custom_emoji
        formatted_note += parsed_entity
    if offset < end:
        formatted_note += from_u16(text[offset:end])
    return formatted_note
#!!!
def is_single_url(message: Message) -> bool:
    # assuming there is atleast one entity
    entities_single_url = message['entities']
    url_entity = entities_single_url[0]
    if url_entity.type == "url":
        return True
    if url_entity.type != "text_link":
        return False
    # need to check nested entities
    url_end = url_entity.offset + url_entity.length
    for e in entities_single_url[1:]:
        if e.offset > url_end:
            return False
    return True
#!!!
async def download(url, session: aiohttp.ClientSession) -> str:
    async with session.get(url) as response:
        return await response.text()
#!!!
def get_open_graph_props(page: str) -> dict:
    props = {}
    soup = BeautifulSoup(page, 'lxml')
    meta = soup.find_all("meta", property=lambda x: x is not None and x.startswith("og:"))
    for m in meta:
        props[m['property'][3:].lstrip()] = m['content']
    if not 'description' in props:
        m = soup.find("meta", attrs={"name": "description"})
        if m:
            props['description'] = m['content']
    if not 'title' in props:
        props['title'] = soup.title.string

    return props
#!!!
async def get_url_info_formatting(url: str) -> str:
    async with aiohttp.ClientSession() as session:
        page = await download(url, session)
        og_props = get_open_graph_props(page)
        if 'image' in og_props or 'description' in og_props:
            sep = ''
            image = ''
            callout_type = "[!link-info-ni]"
            if 'image' in og_props:
                image += "!["
                if 'image:alt' in og_props:
                   image += og_props['image:alt'].replace("\n", " ")
                image += f"]({og_props['image']})"
                if 'image:width' in og_props and int(og_props['image:width']) < 600:
                    callout_type = "[!link-info]"
                else:
                    callout_type = "[!link-preview]"
                sep = "\n>"
            formatted_note = f'\n> {callout_type}'
            if 'site_name' in og_props:
                formatted_note += f" [{og_props['site_name']}]({url})"
            if 'title' in og_props:
                formatted_note += "\n> # " + og_props['title']
            if 'description' in og_props:
                formatted_note += "\n> "
                formatted_note += "\n> ".join(og_props['description'].split('\n')) + sep
            if 'image' in og_props:
                formatted_note += f"\n> [{image}]({url})"
            return formatted_note + "\n"
        return ''
#!!!

async def stt(audio_file_path) -> str:
    import whisper
    print('Whisper imported')
    model = config.whisper_model if 'whisper_model' in dir(config) else 'medium'
    model = whisper.load_model(model)
    print('Model loaded')

    log_basic('Audio recognition started')
    print('Audio recognition started')
    result = model.transcribe(audio_file_path, verbose = False, language = 'ru')
    # Clear GPU memory
    del model
    gc.collect()
    torch.cuda.empty_cache()
    print('CUDA started')

    if hasattr(result['segments'], '__iter__'):
        rawtext = ' '.join([segment['text'].strip() for segment in result['segments']])
        rawtext = re.sub(" +", " ", rawtext)

        alltext = re.sub("([\.\!\?]) ", "\\1\n", rawtext)
        if debug_log:
            log_debug(f'Recognized: {alltext}')
        else:
            log_basic(f'Recognized {len(alltext)} characters')
    else:
        alltext = ""
        log_basic('Nothing recognized')
    return alltext

def unique_filename(file: str, path: str) -> str:
    """Change file name if file already exists"""
    # create target folder if not exist
    if not os.path.exists(path):
        os.makedirs(path)
    # check if file exists
    if not os.path.exists(osPathJoin(path, file)):
        return file
    # get file name and extension
    filename, filext = os.path.splitext(file)
    # get full file path without extension only
    filexx = osPathJoin(path, filename)
    # create incrementing variable
    i = 1
    # determine incremented filename
    while os.path.exists(f'{filexx}_{str(i)}{filext}'):
        # update the incrementing variable
        i += 1
    return f'{filename}_{str(i)}{filext}'

def unique_indexed_filename(file: str, path: str) -> str:
    """Add minimal unique numeric index to file name to make up non existing file name"""
    # create target folder if not exist
    if not os.path.exists(path):
        os.makedirs(path)
    # get file name and extension
    filename, filext = os.path.splitext(file)
    # get full file path without extension only
    filexx = osPathJoin(path, filename)
    # create incrementing variable
    i = 1
    # determine incremented filename
    while os.path.exists(f'{filexx}{i:02}{filext}'):
        # update the incrementing variable
        i += 1
    unique_indexed_filename = f'{filename}{i:02}{filext}'
    # create file to avoid reusing the same file name more than once
    with open(osPathJoin(path, unique_indexed_filename), 'w') as f:
        f.write('')
    return unique_indexed_filename

async def get_contact_data(message: Message) -> str:

    if message.contact.user_id:
        contact_user  = await get_telegram_username(message.contact.user_id)

    frontmatter_body = ''
    for field, value in message.contact:
        if field not in ('vcard', 'user_id'):
            frontmatter_body += f'{field}: {value}\n'

    note_frontmatter = f'''<!-- YAML front matter -->

---
{frontmatter_body}
---
'''

    fname = message.contact.first_name or ''
    lname = message.contact.last_name or ''
    contact_name = f'{fname} {lname}'.strip()

    if message.contact.user_id:
        note_body = f'''<!-- vcard -->
[[{contact_name}]]
Telegram: {contact_user}
```vcard
{message.contact.vcard}
```
'''
    else: 
        note_body = f'''<!-- vcard -->
[[{contact_name}]]
```vcard
{message.contact.vcard}
```
'''

    return note_frontmatter + note_body

async def get_telegram_username(user_id: int) -> str:
    user_info = await bot.get_chat_member(user_id, user_id)
    if 'username' in user_info.user:
        result = f'[@{user_info.user.username}](https://t.me/{user_info.user.username})'
    else:
        fname = user_info.user.first_name or ''
        lname = user_info.user.last_name or ''
        result = f'{fname} {lname}'.strip()

    return result

async def answer_message(message: Message, answer_text: str):
    # Ограничение Telegram - не более 4096 знаков в сообщении
    msg_len_limit = 4000
    if len(answer_text) <= msg_len_limit:
        await message.reply(answer_text)
    else:
        chunks = text_to_chunks(answer_text, msg_len_limit)
        for chunk in chunks:
            try:
                await message.reply(chunk)
            except Exception as e:
                await message.reply(f'🤷‍♂️ {e}')
            time.sleep(0.03)

def bold(text: str) -> str:
    if format_messages():
        return f'**{text}**'
    else:
        return text

def note_from_message(message: Message):
    msg_date = message.date.strftime('%Y-%m-%d')
    msg_time = message.date.strftime('%H:%M:%S')
    note = Note(date=msg_date, time=msg_time)
    return note

def text_to_chunks(text, max_len):
    """ Принимает строку text и делит её на части длиной до max_len. Возвращает список с частями"""
    sentences = [piece.strip() + '.' for piece in text.split('.')]
    texts = []
    chunk = ''

    for sentence in sentences:
        if len(sentence) > max_len or len(chunk + ' ' + sentence) > max_len:
            # Это предложение не влезает в обрабатываемый фрагмент
            if len(chunk) > 0:
                # Если во фрагменте уже что-то есть, сохраним его
                texts.append(chunk.strip(' '))
                chunk = ''
            # Фрагмент пустой, начинаем наполнять
            if len(sentence) > max_len:
                # Если текущее предложение слишком длинное, засунем во фрагмент только, сколько влезет
                words = sentence.split(' ')
                for word in words:
                    if len(chunk + ' ' + word) < max_len:
                        # Это слово влезает в обрабатываемый фрагмент, можно добавлять
                        chunk += ' ' + word
                    else:
                        # Это слово не влезает в обрабатываемый фрагмент
                        texts.append(chunk.strip(' '))
                        chunk = word
            else:
                # Фрагмент был пустой, так что просто засунем предложение в него
                chunk = sentence

        else:
            # Это предложение влезает в обрабатываемый фрагмент, можно добавлять
            chunk += ' ' + sentence
    # Сохраняем последний фрагмент, если в нём что-то есть
    if len(chunk) > 0: texts.append(chunk.strip(' '))
    return texts

def get_location_note(message: Message) -> str:
    lat = message.location.latitude
    lon = message.location.longitude

    location_note = f'''{bold('Latitude')}: {lat}
{bold('Longitude')}: {lon}
[Google maps](https://www.google.com/maps/search/?api=1&query={lat},{lon}), [Yandex maps](https://yandex.ru/maps/?text={lat}%2C{lon}&z=17)
'''
    return location_note

async def send_message_copy(message):
    try:
       # Send a copy of the received message
        await message.send_copy(chat_id=message.chat.id)
    except TypeError:
       # But not all the types is supported to be copied so need to handle it
        await message.answer("Nice try!")   

# Handlers
    
@dp.message(CommandStart())
async def command_start_handler(message: Message) -> None:
    """
    This handler receives messages with `/start` command
    """
    # Most event objects have aliases for API methods that can be called in events' context
    # For example if you want to answer to incoming message you can use `message.answer(...)` alias
    # and the target chat will be passed to :ref:`aiogram.methods.send_message.SendMessage`
    # method automatically or call API method directly via
    # Bot instance: `bot.send_message(chat_id=message.chat.id, ...)`
    log_basic(f'Starting chat with the user @{message.from_user.username} ({message.from_user.first_name} {message.from_user.last_name}, user_id = {message.from_user.id}), chat_id = {message.chat.id} ({message.chat.title})')
    reply_text = f'Hello {message.from_user.full_name}!\n\nI`m a private bot, I save messages from a private Telegram group to Obsidian inbox.\n\nYour Id: {message.from_user.id}\nThis chat Id: {message.chat.id}\n'
    await message.reply(reply_text)

@dp.message(F.photo)
async def handle_photo(message: Message, bot: Bot):
#    if message.chat.id != config.my_chat_id: return
    print  ('PHOTO')
    log_basic(f'Received photo from @{message.from_user.username}')
    log_message(message)
    note = note_from_message(message)
    photo = message.photo[-1]
    log_basic(photo)
    file_name = unique_indexed_filename(create_media_file_name(message, 'pic', 'jpg'), config.photo_path) # or photo.file_id + '.jpg'
    log_basic('file created')
    full_file_name = osPathJoin(config.photo_path, file_name)
    message.reply(f'Сохранил файл под именем {full_file_name}')
    print(f'Saved photo as {file_name}')
    await bot.download(
        message.photo[-1],
        destination=f"{full_file_name}"
    )
    forward_info = get_forward_info(message)
    photo_and_caption = f'{forward_info}![[{file_name}]]\n{await get_formatted_caption(message)}'
    note.text=photo_and_caption
    save_message(note)
    await answer_message(message, f'Сохранил фото под именем {full_file_name}')
  
@dp.message(F.voice)
async def handle_voice_message(message: Message):
#    if message.chat.id != config.my_chat_id: return

    #New code
    log_basic(f'Received voice message from @{message.from_user.username}')   
    log_message(message)   
    note = note_from_message(message)
    print  ('VOICE ')    
    if not config.recognize_voice:
        log_basic(f'Voice recognition is turned OFF')
        return    
    voice = message.voice.file_id
    path = config.voice_path
    file_name = unique_indexed_filename(create_media_file_name(message, 'voice', 'ogg'), config.voice_path)
    full_file_name = osPathJoin(config.voice_path, file_name)
    log_basic(f'Received voice message {file_name} from @{message.from_user.username}')
    log_message(message)
    note = note_from_message(message)
    print(f'Got voice: {file_name}')
    try:     
        file = await bot.get_file(message.voice.file_id)
        await bot.download_file(file.file_path, full_file_name)
        await answer_message(message, f'Сохранил под именем {full_file_name}')
    except Exception as e:
        log_basic(f'Exception: {e}')
        await answer_message(message, f'🤷‍♂️ {e}')
        return
    try:
        await answer_message(message, f'Trying convert to text')
        await bot.send_chat_action(message.chat.id, "typing")
        note_stt = await stt(full_file_name)
        note.text = note_stt
    except Exception as e:
        await answer_message(message, f'🤷‍♂️ {e}')
    save_message(note)   
    try:
        await answer_message(message, note_stt)
    except Exception as e:
        await answer_message(message, f'🤷‍♂️ {e}')

@dp.message(F.audio)
async def handle_voice_message(message: Message):
#    if message.chat.id != config.my_chat_id: return
    #New code
    log_basic(f'Received audio message from @{message.from_user.username}')   
    log_message(message)   
    note = note_from_message(message)
    print  ('AUDIO ')    
    await message.reply(f'File_id is {message.audio.file_id}')
    if not config.recognize_audio:
        log_basic(f'Voice recognition is turned OFF')
        return    
    audio = message.audio.file_id
    path = config.audio_path
    file_name = unique_filename(message.audio.file_name, config.audio_path)
    full_file_name = osPathJoin(config.audio_path, file_name)
    log_basic(f'Received audio message {full_file_name} from @{message.from_user.username}')
    log_message(message)
    note = note_from_message(message)
    print(f'Got audio: {file_name}')
    try:
        file = await bot.get_file(message.audio.file_id)
        await bot.download_file(file.file_path, full_file_name)
        await answer_message(message, f'Сохранил под именем {full_file_name}')
    except Exception as e:
        log_basic(f'Exception: {e}')
        await answer_message(message, f'🤷‍♂️ {e}')
        return
    try:
        await answer_message(message, f'Trying convert to text')
        await bot.send_chat_action(message.chat.id, "typing")
        note_stt = await stt(full_file_name)
        note.text = note_stt
    except Exception as e:
        await answer_message(message, f'🤷‍♂️ {e}')
    save_message(note)  
    try:
        await answer_message(message, note_stt)
    except Exception as e:
        await answer_message(message, f'🤷‍♂️ {e}')
  
@dp.message(F.text)
async def process_text(message: Message):
#    if message.chat.id != config.my_chat_id: return
    print ('TEXT ')
    log_basic('Received text message from @{message.from_user.username}')
    log_message(message)
    note = note_from_message(message)
    message_body = await embed_formatting(message)
    forward_info = get_forward_info(message)
    if (forward_info) and (message_body):
        note.text = forward_info + message_body
    else:
        if (message_body):
            note.text = message_body
    try:
        save_message(note)
    except Exception as e:
        log_basic(f'Exception: {e}')
        await answer_message(message, f'🤷‍♂️ {e}')
        return
    await answer_message(message, f'Сохранил текст') 

@dp.message(F.document)
async def handle_document(message: Message):
#    if message.chat.id != config.my_chat_id: return
    print  ('DOCUMENT ')
    file_name = unique_filename(message.document.file_name, config.photo_path)
    full_file_name = osPathJoin(config.photo_path, file_name)
    log_basic(f'Received document {file_name} from @{message.from_user.username}. CFP= {config.photo_path}, FN={file_name}, Full_file_nane is {full_file_name}')
    log_message(message)
    note = note_from_message(message)
    print(f'Got document: {file_name}')
    try:    
        file = await bot.get_file(message.document.file_id)
        await bot.download_file(file.file_path, full_file_name)
        await answer_message(message, f'Сохранил под именем {file_name}, OPS={os.path.sep} OPSep={os.path.pathsep}')
    except Exception as e:
        log_basic(f'Exception: {e}')
        await answer_message(message, f'🤷‍♂️ {e}')
        return

#todo в случае картинки - вставить ее в заметку как картинку
    if config.recognize_voice and message.document.mime_type.split('/')[0] == 'audio':
    # Если mime type = "audio/*", распознаем речь аналогично ContentType.AUDIO
        await bot.send_chat_action(chat_id=message['from']['id'], action=types.ChatActions.TYPING)

        file_full_path = osPathJoin(config.photo_path, file_name)
        note_stt = await stt(file_full_path)
        try:
            await answer_message(message, note_stt)
        except Exception as e:
            await answer_message(message, f'🤷‍♂️ {e}')
        # Добавляем подпись, если есть, и имя файла
        if message.caption != None:
            file_details = f'{bold(message.caption)} ({file_name})'
        else:
            file_details = bold(file_name)

        note.text = f'{file_details}\n{note_stt}'
        os.remove(file_full_path)
    else:
        forward_info = get_forward_info(message)
        note.text = f'{forward_info}[[{file_name}]]\n{await get_formatted_caption(message)}'

    save_message(note)
    await answer_message(message, f'Сохранил документ')    
    
@dp.message(F.contact)
async def handle_contact(message: Message):
#    if message.chat.id != config.my_chat_id: return
    log_basic(f'Received contact from @{message.from_user.username}')
    log_message(message)
    note = note_from_message(message)
    print(f'Got contact')
    note.text = await get_contact_data(message)
    save_message(note)
    await answer_message(message, f'Сохранил контакт') 
    
@dp.message(F.location)
async def handle_location(message: Message):
#    if message.chat.id != config.my_chat_id: return
    log_basic(f'Received location from @{message.from_user.username}')
    log_message(message)
    print(f'Got location')
    note = note_from_message(message)
    note.text = get_location_note(message)
    save_message(note)
    await answer_message(message, f'Сохранил геоточку')
    
@dp.message(F.animation)
#Не отлажено - не найден пример для проверки
async def handle_animation(message: Message):
#    if message.chat.id != config.my_chat_id: return
    log_message(message)
    file_name = unique_filename(message.document.file_name, config.animation_path)
    log_basic(f'Received animation {file_name} from @{message.from_user.username}')
    print(f'Got animation: {file_name}')
    note = note_from_message(message)
    full_file_name = osPathJoin(config.animation_path, file_name)
    try:      
        file = await bot.get_file(message.document.file_id)
        await bot.download_file(file.file_path, full_file_name)
        await answer_message(message, f'Сохранил под именем {file_name}')
    except Exception as e:
        log_basic(f'Exception: {e}')
        await answer_message(message, f'🤷‍♂️ {e}')
        return
    note.text = f'{get_forward_info(message)}![[{file_name}]]\n{await get_formatted_caption(message)}'
    save_message(note)
    await answer_message(message, f'Сохранил анимацию')

@dp.message(F.video)
async def handle_video(message: Message):
#    if message.chat.id != config.my_chat_id: return
    log_message(message)
    file_name = unique_filename(message.video.file_name, config.video_path)
    log_basic(f'Received video {file_name} from @{message.from_user.username}')
    print(f'Got video: {file_name}')
    note = note_from_message(message)
    full_file_name = osPathJoin(config.video_path, file_name)
    try:       
        file = await bot.get_file(message.video.file_id)
        await bot.download_file(file.file_path, full_file_name)
        await answer_message(message, f'Сохранил под именем {file_name}')
    except Exception as e:
        log_basic(f'Exception: {e}')
        await answer_message(message, f'🤷‍♂️ {e}')
        return
    note.text = f'{get_forward_info(message)}![[{file_name}]]\n{await get_formatted_caption(message)}'
    save_message(note)
    await answer_message(message, f'Сохранил видео')

@dp.message(F.video_note)
async def handle_video_note(message: Message):
#    if message.chat.id != config.my_chat_id: return
    log_message(message)
    file_name = unique_indexed_filename(create_media_file_name(message, 'video', 'mp4'), config.photo_path)
    log_basic(f'Received video note from @{message.from_user.username}')
    print(f'Got video note: {file_name}')
    note = note_from_message(message)
    full_file_name = osPathJoin(config.video_path, file_name)
    try:    
        file = await bot.get_file(message.video_note.file_id)
        await bot.download_file(file.file_path, full_file_name)
        await answer_message(message, f'Сохранил под именем {file_name}')
    except Exception as e:
        log_basic(f'Exception: {e}')
        await answer_message(message, f'🤷‍♂️ {e}')
        return
    note.text = f'{get_forward_info(message)}![[{file_name}]]\n{await get_formatted_caption(message)}'
    save_message(note)
    await answer_message(message, f'Сохранил видеозаметку')
    
@dp.message()
async def process_message(message: types.Message):
    print(f'Got message of unkonwn type - {message.content_type}')
    await message.reply (f'Unknown type of message - {message.content_type}')
#    if message.chat.id != config.my_chat_id: return
    print('Message processing')
    await send_message_copy(message)
    log_basic('Received text message from @{message.from_user.username}')
    log_message(message)
    note = note_from_message(message)
    message_body = await embed_formatting(message)
 # Here is the problem with small pictures and polls
    forward_info = get_forward_info(message)
    if (forward_info) and (message_body):
        note.text = forward_info + message_body
    else:
        if (message_body):
            note.text = message_body
    try:
        save_message(note)
    except Exception as e:
        log_basic(f'Exception: {e}')
        await answer_message(message, f'🤷‍♂️ {e}')
        return

# Запуск процесса поллинга новых апдейтов

async def main() -> None:
    # Initialize Bot instance with a default parse mode which will be passed to all API calls
    # And the run events dispatching
    await dp.start_polling(bot)


if __name__ == "__main__":
    print('Bot started')
    log.info ('Bot started')
    asyncio.run(main())

import io
import os

from telegram.ext import CommandHandler
from telegram.ext import Updater, MessageHandler, Filters

from model import get_prediction_by

telegram_bot_token = os.getenv("TOKEN")

updater = Updater(token=telegram_bot_token, use_context=True)
dispatcher = updater.dispatcher


def start(update, context):
    """
    Set up the introductory statement for the bot when the /start command is invoked
    """
    chat_id = update.effective_chat.id
    greeting_message = "Приветствую! Приложите фотографию собаки, породу которой хотите классифицировать"
    message_note = "Пока я умею классифицировать только такие породы:\nShih-Tzu\nRhodesian ridgeback\nBeagle\n" \
                   "English foxhound\nBorder terrier\nAustralian terrier\nGolden retriever\nOld English sheepdog\n" \
                   "Samoyed\nDingo"
    context.bot.send_message(chat_id=chat_id,
                             text=greeting_message + "\n\n" + message_note)


def full_photo_input_handler(update, context):
    """
    Handle case when user upload document into bot
    """
    file = context.bot.get_file(update.message.document)
    file = io.BytesIO(file.download_as_bytearray())
    dog, prob = get_prediction_by(file)
    prob = "{:.1f}".format(prob * 100)
    message = f"Порода собаки: {dog}\nВероятность: {prob}%\n"
    update.message.reply_text(message)


def compressed_photo_input_handler(update, context):
    """
    Handle case when user upload photo into bot
    """
    file = context.bot.get_file(update.message.photo[-1].file_id)
    file = io.BytesIO(file.download_as_bytearray())
    dog, prob = get_prediction_by(file)
    prob = "{:.1f}".format(prob * 100)
    message = f"Порода собаки: {dog}\nВероятность: {prob}%\n"
    update.message.reply_text(message)


def text_input_handler(update, context):
    """
    Handle case when user text something to bot except for uploading photo
    """
    message = "Я не понимаю слова, я понимаю только язык картинок с собачками, прикрепите пожалуйста файл с собакой"
    update.message.reply_text(message)


# run the start function when the user invokes the /start command
dispatcher.add_handler(CommandHandler("start", start))

dispatcher.add_handler(MessageHandler(Filters.text, text_input_handler))
dispatcher.add_handler(MessageHandler(Filters.photo, compressed_photo_input_handler))
dispatcher.add_handler(MessageHandler(Filters.document, full_photo_input_handler))

updater.start_webhook(listen="0.0.0.0",
                      port=int(os.environ.get('PORT', 5000)),
                      url_path=telegram_bot_token,
                      webhook_url='https://dog-breed-classifier1.herokuapp.com/' + telegram_bot_token
                      )

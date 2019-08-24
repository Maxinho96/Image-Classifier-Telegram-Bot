from telegram.ext import Updater, MessageHandler, Filters
import os
import neural_network as nn
import utils

net = nn.NeuralNetwork()
util = Utils()

def classify_image(bot, update):
    image_file = bot.getFile(update.message.photo[-1].file_id)
    image_file.download("image.jpg")
    preds = net.get_predictions("image.jpg")
    text = util.preds_to_string(preds)
    update.message.reply_markdown(text)

def main():
    TOKEN = os.getenv("TOKEN")

    updater = Updater(TOKEN)
    dp = updater.dispatcher

    dp.add_handler(MessageHandler(Filters.photo, classify_image))

    PORT = int(os.environ.get("PORT", "8443"))
    HEROKU_APP_NAME = os.environ.get("HEROKU_APP_NAME")
    updater.start_webhook(listen="0.0.0.0", port=PORT, url_path=TOKEN)
    updater.bot.set_webhook("https://{}.herokuapp.com/{}".format(HEROKU_APP_NAME, TOKEN))


if __name__ == '__main__':
    main()

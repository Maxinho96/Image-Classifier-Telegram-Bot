from telegram.ext import Updater, MessageHandler, Filters
import os
import neural_network as nn
import utils
from functools import partial

def classify_image(bot, update, net, util):
    image_file = bot.getFile(update.message.photo[-1].file_id)
    image_file.download("image.jpg")
    preds = net.get_predictions("image.jpg")
    text = util.preds_to_string(preds)
    update.message.reply_markdown(text)

def main():
    TOKEN = os.getenv("TOKEN")

    updater = Updater(TOKEN)
    dp = updater.dispatcher

    net = nn.NeuralNetwork()
    util = utils.Utils()
    classify_image_callback = partial(classify_image, net=net, util=util)

    dp.add_handler(MessageHandler(Filters.photo, classify_image_callback))

    PORT = int(os.environ.get("PORT", "8443"))
    HEROKU_APP_NAME = os.environ.get("HEROKU_APP_NAME")
    updater.start_webhook(listen="0.0.0.0", port=PORT, url_path=TOKEN)
    updater.bot.set_webhook("https://{}.herokuapp.com/{}".format(HEROKU_APP_NAME, TOKEN))


if __name__ == '__main__':
    main()

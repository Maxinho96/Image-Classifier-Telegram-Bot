#!/bin/bash

heroku container:push --app imageclassifierbot web
heroku container:release --app imageclassifierbot web

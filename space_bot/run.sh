#!/usr/bin/with-contenv bashio
set -e

export DISCORD_BOT_TOKEN=$(bashio::config 'discord_bot_token')
export NASA_API_KEY=$(bashio::config 'nasa_api_key')
export FLICKR_API_KEY=$(bashio::config 'flickr_api_key' '')
export LOG_LEVEL=$(bashio::config 'log_level' 'INFO')

if [ -z "${DISCORD_BOT_TOKEN}" ]; then
    bashio::log.fatal "discord_bot_token is not set in the add-on configuration."
    exit 1
fi

if [ -z "${FLICKR_API_KEY}" ]; then
    bashio::log.warning "flickr_api_key is not set - ESA and SpaceX channels will be skipped (get a free key at https://www.flickr.com/services/apps/create/apply)."
fi

bashio::log.info "Starting Space Image Discord Bot..."
cd /data
exec python3 /app/space_bot.py
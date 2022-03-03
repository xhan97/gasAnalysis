# Copyright 2022 Xin Han
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from email.policy import default
import logging
import os
from pathlib import Path

import click
from dotenv import find_dotenv, load_dotenv

from price.price import make_price
from weather.ecmen import Ecmen
from weather.ecmop import Ecmop
from weather.gfsen import Gfsen
from weather.gfsop import Gfsop


def make_weather(weather_name, weather_path, st_date, out_path):
    if weather_name == 'ecmen':
        weather = Ecmen(path=weather_path)
    elif weather_name == 'ecmop':
        weather = Ecmop(path=weather_path)
    elif weather_name == 'gfsop':
        weather = Gfsop(path=weather_path)
    elif weather_name == 'gfsen':
        weather = Gfsen(path=weather_path)
    else:
        raise NotImplementedError
    weather.load_data(start_date=st_date).merge_data.transform_dst.get_delta.get_df(
        save=os.path.join(out_path, weather_name+"_weather_subclass.csv"))


@click.command()
@click.argument('archive_input_path', default='data/raw/Archive', type=click.Path(exists=True))
@click.argument('weather_name', default='ecmen')
@click.argument('weather_input_path', default='data/raw/WeatherData/ECMEN_WDD_Forecasts_20100101_20210331.csv.gz', type=click.Path(exists=True))
@click.argument('archive_output_path', default='data/processed/Archive', type=click.Path())
@click.argument('weather_output_path', default='data/processed/WeatherData', type=click.Path())
@click.argument('start_year', default='2015')
def main(start_year, archive_input_path, archive_output_path, weather_name, weather_input_path, weather_output_path):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('making price data from raw data')
    os.makedirs(archive_output_path, exist_ok=True)
    os.makedirs(weather_output_path, exist_ok=True)
    make_price(archive_input_path, archive_output_path, start_year)
    logger.info('making weather data from raw data')
    make_weather(weather_name=weather_name, weather_path=weather_input_path,
                 st_date=start_year, out_path=weather_output_path)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()

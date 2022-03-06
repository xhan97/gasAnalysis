# Copyright 2021 Xin Han
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

from genericpath import exists
import logging
import os
from email.policy import default
from pathlib import Path

import click
import pandas as pd
from dotenv import find_dotenv, load_dotenv

from period_weather import make_period_cutoff


@click.command()
@click.argument('weather_name', default='ecmen')
@click.argument('weather_path', default='data/processed/WeatherData/ecmen_weather_subclass.csv', type=click.Path(exists=True))
@click.argument('cutoff_path', default='data/processed/Archive/cut_off_price.csv', type=click.Path(exists=True))
@click.argument('start_time', default='6:00')
@click.argument('end_time', default='16:00')
@click.argument('using_period', default=1, type=click.INT)
@click.argument('output_dir', default='data/processed/period', type=click.Path())
def main(weather_name, weather_path, cutoff_path, start_time, end_time, using_period, output_dir):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('building features')
    os.makedirs(output_dir, exist_ok=True)
    make_period_cutoff(weather_name=weather_name, weather_path=weather_path, cutoff_path=cutoff_path,
                       st_time=start_time, ed_time=end_time, out_dir=output_dir, using_period=using_period)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()

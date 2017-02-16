#!bin/bash/

# Script for loading data incase something goes wrong
# ---------------------------------------------------
# First, export env variables and then run script
# Takes a minute, because data is around 3.5 gb
# This can be used to load to local db or Heroku

python convert_to_csv.py ${PATH_TO_FILE} ${PATH_TO_OUTPUT}
PGPASSWORD=${PGPW} psql -h ${HOST} -U ${USER} ${DB} < import.sql


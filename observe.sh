#!/usr/bin/bash

function run () {

SAVE="platoon_simplified"

python3 observe.py -d save/$SAVE/DuelingDoubleDQNAgent_lr0.0002_model.pack

}

cd ..

source venv/bin/activate

run

deactivate

exit

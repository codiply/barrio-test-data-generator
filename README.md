# Barrio test data generator

## Movie Lens item factors

First download the data

    python .\scripts\download-movie-lens-data.py

Then you need to increase your sbt memory

    SBT_OPTS = "-Xmx4G"
    
and run

    sbt run
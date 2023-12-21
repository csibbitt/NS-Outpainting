CHECKPOINT=24963

DATAHASH="126nsDuvSPR-NYA3x9a4eJr92HQrGrGm_"
INDEXHASH="129t1wcxzmnpxt36EZr4lc-JbVlPdYEHm"
METAHASH="129eYo2w3Z5PC2BoMLgXHMUT7RrVjQ-1Z"

wget "https://drive.google.com/uc?export=download&id=${DATAHASH}&confirm=yes" -O "logs/1215/2/models/-${CHECKPOINT}.data-00000-of-00001"
wget "https://drive.google.com/uc?export=download&id=${INDEXHASH}&confirm=yes" -O "logs/1215/2/models/-${CHECKPOINT}.index"
wget "https://drive.google.com/uc?export=download&id=${METAHASH}&confirm=yes" -O "logs/1215/2/models/-${CHECKPOINT}.meta"

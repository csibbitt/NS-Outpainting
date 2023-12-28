CHECKPOINT=30458

DATAHASH="16TBkuePRqzhdHPOFTaOLvmt4OxXboiNb"
INDEXHASH="16pTciNWMQ7u13DItblUEmjeQfhTV5hUN"
METAHASH="16hcLDmRQFdG0QsIb4Su7jqCSUsmEGioW"

wget "https://drive.google.com/uc?export=download&id=${DATAHASH}&confirm=yes" -O "models/-${CHECKPOINT}.data-00000-of-00001"
wget "https://drive.google.com/uc?export=download&id=${INDEXHASH}&confirm=yes" -O "models/-${CHECKPOINT}.index"
wget "https://drive.google.com/uc?export=download&id=${METAHASH}&confirm=yes" -O "models/-${CHECKPOINT}.meta"

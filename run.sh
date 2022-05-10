python src/feature/msglog_feature.py
echo ">> msg feature process end"

python src/feature/venus_feature.py
echo ">> venus feature process end"

python src/feature/crashdump_feature.py
echo ">> crashdump feature process end"

python src/feature/feature_merge.py
echo ">> featuremerge process end" 

python src/main.py
echo ">> training end"

python src/submit.py
zip -j result.zip submission/submit.csv
echo ">> submit end"

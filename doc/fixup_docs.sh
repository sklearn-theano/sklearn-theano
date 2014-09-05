# Github pages ignore anything starting with an _
# Lovely
pushd . 
cd _build/html/dev
#for folder in static downloads images sources; do
for folder in static downloads images sources; do
    mv _$folder $folder
    grep -rl _$folder * | xargs sed -i s:_$folder/:$folder/:g
done
popd

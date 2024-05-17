pwd=$(pwd -P)
lwd=$(dirname $pwd)

echo ${proto_dir}
for item in "$proto_dir"/*; do
    filename=$(basename $item)
    if [[ $filename =~ \.proto$ ]]; then
        protoc -I $proto_dir $item  --python_out=$proto_dir
    fi
done


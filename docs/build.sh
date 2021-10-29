for nb in *.ipynb
do
    jupyter nbconvert --to pdf --execute $nb
done

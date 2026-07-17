.PHONY: install eval test figures notebook clean

install:
	pip install -r requirements.txt

eval:
	python scripts/run_eval.py

figures:
	python scripts/make_figures.py

test:
	PYTHONPATH=src pytest -q

notebook:
	jupyter nbconvert --to notebook --execute --inplace notebooks/analysis.ipynb

clean:
	rm -rf results/results.csv results/figures/*.png

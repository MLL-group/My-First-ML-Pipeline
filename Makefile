setup:
	curl -sSL https://install.python-poetry.org | python3 -
install-deps:
	poetry install --no-root --sync

exercise-1:
	poetry run python eda.py

exercise-2:
	poetry run python simple_model.py

exercise-3:
	poetry run python prep_dataset.py

exercise-4:
	poetry run python modeling.py

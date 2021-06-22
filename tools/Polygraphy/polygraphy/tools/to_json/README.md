# To-JSON

This is a temporary tool that exists to convert old pickled data from Polygraphy to JSON.
In 0.27.0, Polygraphy migrated to JSON serialization for security reasons.

## Usage

Simply provide the tool with a path to your old pickled data, and an output path:
```bash
polygraphy to-json old_data.pkl -o data.json
```

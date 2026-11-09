from fastapi import FastAPI, HTTPException
import uvicorn


app = FastAPI()


@app.get('/')
def home():
    return {'message': 'Hello World!'}


@app.post('/')
def post():
    return {'message': 'POST Request'}


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)

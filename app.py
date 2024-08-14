from typing import Optional
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from uvicorn import run as app_run


from insurance_structure.constant.application import APP_HOST, APP_PORT
from insurance_structure.pipeline.prediction_pipline import HeartData, HeartStrokeClassifier
from insurance_structure.pipeline.train_pipeline import TrainPipeline
from insurance_structure.pipeline.prediction_pipline import HeartData

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

templates = Jinja2Templates(directory='templates')

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class DataForm:
    def __init__(self, request: Request):
        self.request: Request = request
        self.sex: Optional[str] = None
        self.age: Optional[int] = None
        self.bmi: Optional[float] = None
        self.children: Optional[int] = None
        self.smoker: Optional[str] = None
        self.region: Optional[str] = None
        self.charges: Optional[float] = None

    async def get_stroke_data(self):
        form = await self.request.form()
        
        # Helper function to safely convert strings to floats
        def safe_float(value: str, default: float) -> float:
            try:
                return float(value) if value else default
            except ValueError:
                return default
        
        # Helper function to safely convert strings to integers
        def safe_int(value: str, default: int) -> int:
            try:
                return int(value) if value else default
            except ValueError:
                return default
        
        self.sex = form.get("sex")
        self.age = safe_int(form.get("age", "0"), 0)  # Default to 0 if not provided
        self.bmi = safe_float(form.get("bmi", "0.0"), 0.0)  # Default to 0.0 if not provided
        self.children = safe_int(form.get("children", "0"), 0)  # Default to 0 if not provided
        self.smoker = form.get("smoker")
        self.region = form.get("region")
        self.charges = safe_float(form.get("charges", "0.0"), 0.0)  # Default to 0.0 if not provided

@app.get("/", tags=["authentication"])
async def index(request: Request):
    return templates.TemplateResponse(
        "index.html", {"request": request, "context": "Rendering"}
    )

@app.get("/train")
async def trainRouteClient():
    try:
        train_pipeline = TrainPipeline()
        train_pipeline.run_pipeline()
        return Response("Training successful !!")
    except Exception as e:
        return Response(f"Error Occurred! {e}")

@app.post("/")
async def predictRouteClient(request: Request):
    try:
        form = DataForm(request)
        await form.get_stroke_data()
        
        # Create InsuranceData instance with the form data
        insurance_data = HeartData(
            sex=form.sex,
            age=form.age,
            bmi=form.bmi,
            children=form.children,
            smoker=form.smoker,
            region=form.region,
            charges=form.charges
        )
        
        # Convert the InsuranceData instance to DataFrame using the correct method
        insurance_data_df = insurance_data.get_heart_stroke_input_data_frame()  # Correct method name
        
        # Predict using the InsuranceRegression model
        model_predictor = HeartStrokeClassifier()
        prediction_result = model_predictor.predict(dataframe=insurance_data_df)

        return templates.TemplateResponse(
            "index.html",
            {"request": request, "context": prediction_result},
        )
        
    except Exception as e:
        return {"status": False, "error": f"{e}"}



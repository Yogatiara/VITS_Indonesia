from fastapi import APIRouter, Form, status
from utils.audio_inference import create_audio

router = APIRouter()



@router.post('/inference', status_code=status.HTTP_201_CREATED)
async def inference(small_model: bool = Form(False),
                    text: str = Form(...),
                    DP_adversarial_learning : bool = Form(...), 
                    SDP: bool = Form(...)
                    ):
  audio = create_audio(small_model=small_model, text=text, DP_adversarial_learning=DP_adversarial_learning, SDP=SDP)
  
  return {
    "message": "Successfull Inference",
    "data": {
        **audio
      }   
    }

@router.get('/get-audio', status_code=status.HTTP_200_OK)
async def inference():
  return {"message": "Hello World"}
import pydantic
from numpydantic import NDArray


class ImageMaskOrPrediction(pydantic.BaseModel):

    id : str = pydantic.Field(
        frozen = True
    )
    path : pydantic.FilePath | None = pydantic.Field(
        frozen = True
    )
    shape : tuple | None = pydantic.Field(
        description = "h,w,c format",
        default = (512, 512, 3)
    )
    encoded_string : str | None = pydantic.Field(
        default = None
    )


class Datapoint(pydantic.BaseModel):

    id : str =  pydantic.Field(
        frozen = True
    )
    uuid : str | None = pydantic.Field()    
    source : str | int | None = pydantic.Field()
    image : ImageMaskOrPrediction = pydantic.Field()
    mask : ImageMaskOrPrediction | None = pydantic.Field(
        default = None
    )
    predicted_mask : ImageMaskOrPrediction | None = pydantic.Field(
        default = None
    )
    

class Dataset(pydantic.BaseModel):

    name : str | None = pydantic.Field(
        default = None
    )

    role : str | None = pydantic.Field(
        default = "train"
    )

    datapoints :  NDArray | list | None = pydantic.Field(
        default = []
    )

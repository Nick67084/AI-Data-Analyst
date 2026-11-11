from typing import List

from pydantic import BaseModel, Field


class ColumnProfile(BaseModel):
    name: str
    dtype: str
    detected_type: str
    analytical_role: str
    null_count: int = Field(ge=0)
    null_percentage: float = Field(
        ge=0,
        le=100,
    )
    unique_count: int = Field(ge=0)


class TableProfile(BaseModel):
    table_name: str
    rows: int = Field(ge=0)
    columns: int = Field(ge=0)
    duplicate_rows: int = Field(ge=0)
    column_details: List[ColumnProfile]


class DatasetProfile(BaseModel):
    filename: str
    file_type: str
    tables: List[TableProfile]
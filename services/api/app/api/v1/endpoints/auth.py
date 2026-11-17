from app.core.deps import CurrentUser, get_current_user
from fastapi import APIRouter, Depends, status
from pydantic import BaseModel, EmailStr

router = APIRouter(prefix="/auth", tags=["Auth"])


class UserResponse(BaseModel):
    id: str
    email: str
    display_name: str
    role: str = "user"


class AuthResponse(BaseModel):
    user: UserResponse
    access_token: str
    refresh_token: str


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str
    display_name: str


class RefreshRequest(BaseModel):
    refresh_token: str


class WsTicketResponse(BaseModel):
    ticket: str
    expires_in: int = 60


@router.post("/login", response_model=AuthResponse)
async def login(req: LoginRequest) -> AuthResponse:
    user = UserResponse(
        id="00000000-0000-7000-8000-000000000001",
        email=req.email,
        display_name="Demo User",
        role="admin",
    )
    return AuthResponse(
        user=user,
        access_token="mock_access_token_c0",
        refresh_token="mock_refresh_token_c0",
    )


@router.post("/register", response_model=AuthResponse, status_code=status.HTTP_201_CREATED)
async def register(req: RegisterRequest) -> AuthResponse:
    user = UserResponse(
        id="00000000-0000-7000-8000-000000000001",
        email=req.email,
        display_name=req.display_name,
        role="user",
    )
    return AuthResponse(
        user=user,
        access_token="mock_access_token_c0",
        refresh_token="mock_refresh_token_c0",
    )


@router.get("/me", response_model=UserResponse)
async def me(current_user: CurrentUser = Depends(get_current_user)) -> UserResponse:
    return UserResponse(
        id=current_user.id,
        email=current_user.email,
        display_name="Demo User",
        role=current_user.role,
    )


@router.post("/ws-ticket", response_model=WsTicketResponse, status_code=status.HTTP_201_CREATED)
async def get_ws_ticket(current_user: CurrentUser = Depends(get_current_user)) -> WsTicketResponse:
    return WsTicketResponse(
        ticket="ws_ticket_mock_abc123",
        expires_in=60,
    )

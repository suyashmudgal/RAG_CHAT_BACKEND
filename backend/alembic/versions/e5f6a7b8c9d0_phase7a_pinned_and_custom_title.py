"""phase7a_pinned_and_custom_title

Revision ID: e5f6a7b8c9d0
Revises: d4e5f6a7b8c9
Create Date: 2026-09-17 12:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'e5f6a7b8c9d0'
down_revision: Union[str, Sequence[str], None] = 'd4e5f6a7b8c9'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Add is_pinned and is_custom_title columns, backfill, and composite index."""
    # 1. Add columns to conversations
    op.add_column(
        'conversations',
        sa.Column('is_pinned', sa.Boolean(), nullable=False, server_default=sa.text('false'))
    )
    op.add_column(
        'conversations',
        sa.Column('is_custom_title', sa.Boolean(), nullable=False, server_default=sa.text('false'))
    )

    # 2. Backfill existing conversations
    op.execute("UPDATE conversations SET is_pinned = false WHERE is_pinned IS NULL")
    op.execute("UPDATE conversations SET is_custom_title = true WHERE title != 'New Conversation'")

    # 3. Add composite index for pinned + updated_at ordering
    op.create_index(
        'ix_conversations_user_pin_updated',
        'conversations',
        ['user_id', sa.text('is_pinned DESC'), sa.text('updated_at DESC')],
        unique=False
    )


def downgrade() -> None:
    """Revert phase7a migration."""
    op.drop_index('ix_conversations_user_pin_updated', table_name='conversations')
    op.drop_column('conversations', 'is_custom_title')
    op.drop_column('conversations', 'is_pinned')

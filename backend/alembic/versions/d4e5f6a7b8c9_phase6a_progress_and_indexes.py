"""phase6a_progress_and_indexes

Revision ID: d4e5f6a7b8c9
Revises: 9ffbbc8ff6d7
Create Date: 2026-09-16 13:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'd4e5f6a7b8c9'
down_revision: Union[str, Sequence[str], None] = '9ffbbc8ff6d7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema: add progress tracking columns and performance composite indexes."""
    # 1. Add progress tracking columns to documents
    op.add_column(
        'documents',
        sa.Column('progress', sa.Integer(), nullable=False, server_default=sa.text('0'))
    )
    op.add_column(
        'documents',
        sa.Column('processing_stage', sa.String(length=50), nullable=True, server_default=sa.text("'COMPLETED'"))
    )
    op.add_column(
        'documents',
        sa.Column('processed_chunks', sa.Integer(), nullable=False, server_default=sa.text('0'))
    )
    op.add_column(
        'documents',
        sa.Column('error_message', sa.Text(), nullable=True)
    )

    # 2. Backfill existing processed documents so they have 100% progress and COMPLETED stage
    op.execute(
        "UPDATE documents SET processing_stage = 'COMPLETED', progress = 100, processed_chunks = chunk_count "
        "WHERE status = 'processed' OR status = 'COMPLETED'"
    )

    # 3. Add composite indexes for high performance querying
    op.create_index(
        'ix_conversations_user_id_updated_at',
        'conversations',
        ['user_id', sa.text('updated_at DESC')],
        unique=False
    )
    op.create_index(
        'ix_messages_conversation_id_created_at',
        'messages',
        ['conversation_id', sa.text('created_at ASC')],
        unique=False
    )
    op.create_index(
        'ix_documents_user_id_created_at',
        'documents',
        ['user_id', sa.text('created_at DESC')],
        unique=False
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index('ix_documents_user_id_created_at', table_name='documents')
    op.drop_index('ix_messages_conversation_id_created_at', table_name='messages')
    op.drop_index('ix_conversations_user_id_updated_at', table_name='conversations')

    op.drop_column('documents', 'error_message')
    op.drop_column('documents', 'processed_chunks')
    op.drop_column('documents', 'processing_stage')
    op.drop_column('documents', 'progress')

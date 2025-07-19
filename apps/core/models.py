"""
Core models and abstract base classes.
"""
import uuid
from django.db import models
from django.utils import timezone


class TimestampedModel(models.Model):
    """Abstract base class with timestamp fields."""
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        abstract = True


class UUIDModel(models.Model):
    """Abstract base class with UUID primary key."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    
    class Meta:
        abstract = True


class BaseModel(TimestampedModel, UUIDModel):
    """Base model combining timestamp and UUID functionality."""
    
    class Meta:
        abstract = True


class SoftDeleteManager(models.Manager):
    """Manager that excludes soft-deleted objects."""
    
    def get_queryset(self):
        return super().get_queryset().filter(deleted_at__isnull=True)


class SoftDeleteModel(BaseModel):
    """Abstract model with soft delete functionality."""
    deleted_at = models.DateTimeField(null=True, blank=True)
    
    objects = SoftDeleteManager()
    all_objects = models.Manager()  # Access to all objects including deleted
    
    class Meta:
        abstract = True
    
    def delete(self, using=None, keep_parents=False):
        """Soft delete the object."""
        self.deleted_at = timezone.now()
        self.save(using=using)
    
    def hard_delete(self, using=None, keep_parents=False):
        """Permanently delete the object."""
        super().delete(using=using, keep_parents=keep_parents)
    
    def restore(self):
        """Restore a soft-deleted object."""
        self.deleted_at = None
        self.save()
    
    @property
    def is_deleted(self):
        return self.deleted_at is not None
class ProtonException(Exception):
    def __init__(
        self,
        message: str | None = None,
    ):
        super(ProtonException, self).__init__(message)

class InvalidArgsError(ProtonException):
    """A proton method was called with invalid arguments."""

class InvalidContext(ProtonException):
    """Incorrect sequence of API calls."""

class InferenceThreadTimeout(ProtonException):
    """Inference thread failed to complete within configured timeout."""

class BadPredictionError(ProtonException):
    """Inference result is not acceptable."""

class Http429RateLimitError(ProtonException):
    """API rate limit."""

class Http500ServerError(ProtonException):
    """API rate limit."""

class AbstractMethodNotImplementedError(ProtonException):
    """User did not provide an implementation for a required method."""

class FileNotFoundError(ProtonException):
    """File not found."""

class DirectoryNotExistError(ProtonException):
    """ Directory does not exist """

class InvalidApiResponseError(ProtonException):
    """API response is missing important information."""

class InferenceApiTimeout(ProtonException):
    """Inference API failed to complete within configured timeout."""

class DataValueError(ProtonException):
    """Data value is not acceptable."""

class AsposePlatformNotSupportedError(ProtonException):
   """Aspose import error"""

class AsposeLicenseError(ProtonException):
   """Aspose license error"""

class AsposeLicenseExpiredError(ProtonException):
   """Aspose license expired"""

class AsposeLicenseInvalidError(ProtonException):
   """Aspose license invalid"""

class AsposeLicenseNotAvailableError(ProtonException):
   """Aspose license not available"""

class AsposeExtractionError(ProtonException):
   """Aspose extraction error"""

class ListIndexOutOfRangeError(ProtonException):
   """List index out of range"""

class UnsupportedFileTypeError(ProtonException):
   """Unsupported file type"""

class UnsupportedFileFormatError(ProtonException)   :
   """Unsupported file format"""

class DocumentAIExtractorValueError(ProtonException):
    """Document AI Extractor Value error"""

class DataIngestionError(ProtonException):
    """ Data Ingestion Service Error """

class DataLoaderExtractionTaskError(ProtonException):
    """ Dataloader Extraction Error """

class BaseFileStoreIOError(ProtonException):
    """ Base File Store IO Error """

class KenshoExtractionError(ProtonException):
    """ Kensho Extract Extraction Error """

class KenshoExtractionPageTooShortException(ProtonException):
    """Too many lines for the specified page size""" 



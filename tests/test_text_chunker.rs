use py2rs::text_chunker;

#[test]
fn test_read_text_file() {
    let document_uri = "books/book1.txt"; 
    let expected_output = "A text chunker breaks large bodies of text into smaller. that are manageable pieces."; 

    match text_chunker::read_text_file(document_uri, true) {
        Ok(content) => assert_eq!(content, expected_output),
        Err(e) => panic!("Failed to read text file: {}", e),
    }
}

#[test]
fn test_fix_line_breaks() {
    let original_txt = "First line.\nSecond line.\rThird line.";
}

#[test]
fn fix_apostrophes() {
    
}

#[test]
fn test_normalize_special_characters() {

}
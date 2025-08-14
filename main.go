package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/gen2brain/go-fitz" // PDF processing
	"github.com/joho/godotenv"
	openai "github.com/sashabaranov/go-openai" // OpenAI SDK
)

// ==============================
// CONFIGURATION
// ==============================
const (
	resultsFile   = "summarized_results.json"
	pdfsDir       = "PDF"
	chunkSize     = 3 // Process files in chunks of 3
	maxTextLength = 3000
)

// ==============================
// DATA STRUCTURES
// ==============================
type ArticleSummary struct {
	Title       string   `json:"title"`
	Objectives  string   `json:"objectives"`
	StudyType   string   `json:"study_type"`
	Methodology string   `json:"methodology"`
	Findings    string   `json:"findings"`
	Conclusions string   `json:"conclusions"`
	Limitations string   `json:"limitations"`
	Keywords    []string `json:"keywords"`
}

type SubjectGroup struct {
	Subject      string           `json:"subject"`
	Articles     []ArticleSummary `json:"articles"`
	GroupSummary string           `json:"group_summary"`
}

type Results struct {
	Groups []SubjectGroup `json:"subject_groups"`
}

func loadConfig() {
	// Carrega o arquivo .env
	err := godotenv.Load()
	if err != nil {
		log.Fatal("Error loading .env file")
	}

	// Validação das variáveis
	requiredVars := []string{"OPENAI_API_KEY", "RESULTS_FILE", "PDFS_DIR"}
	for _, v := range requiredVars {
		if os.Getenv(v) == "" {
			log.Fatalf("Variável %s não encontrada no .env", v)
		}
	}
}

// ==============================
// MAIN FUNCTION
// ==============================
func main() {
	rand.Seed(time.Now().UnixNano())
	loadConfig()

	// Get all PDF files
	pdfFiles, err := getPDFFiles(pdfsDir)
	if err != nil {
		log.Fatalf("Error getting PDF files: %v", err)
	}

	if len(pdfFiles) < chunkSize {
		log.Fatalf("Directory needs at least %d PDF files", chunkSize)
	}

	// Shuffle and process in chunks
	rand.Shuffle(len(pdfFiles), func(i, j int) {
		pdfFiles[i], pdfFiles[j] = pdfFiles[j], pdfFiles[i]
	})

	var results Results

	for i := 0; i < len(pdfFiles); i += chunkSize {
		end := i + chunkSize
		if end > len(pdfFiles) {
			end = len(pdfFiles)
		}

		chunk := pdfFiles[i:end]
		groupName := fmt.Sprintf("Group_%d", (i/chunkSize)+1)

		fmt.Printf("\n📂 Processing %s (%d files):\n", groupName, len(chunk))
		for _, file := range chunk {
			fmt.Printf("- %s\n", filepath.Base(file))
		}

		// Process the chunk
		subject, summaries, groupSummary, err := processPDFGroup(chunk)
		if err != nil {
			log.Printf("Error processing group: %v", err)
			continue
		}

		results.Groups = append(results.Groups, SubjectGroup{
			Subject:      subject,
			Articles:     summaries,
			GroupSummary: groupSummary,
		})
	}

	// Save structured results
	if err := saveStructuredResults(results); err != nil {
		log.Fatalf("Error saving results: %v", err)
	}

	fmt.Println("\n✅ Processing complete. Results saved to", resultsFile)
}

// ==============================
// PROCESSING FUNCTIONS
// ==============================
func processPDFGroup(files []string) (string, []ArticleSummary, string, error) {
	client := openai.NewClient(os.Getenv("OPENAI_API_KEY"))

	// First pass: Extract subject from filenames
	subject := determineSubject(files)

	// Process each article
	var summaries []ArticleSummary
	var combinedText strings.Builder

	for _, file := range files {
		text, err := extractTextFromPDF(file)
		if err != nil {
			return "", nil, "", fmt.Errorf("error reading %s: %v", file, err)
		}

		// Get structured summary for each article
		summary, err := getArticleSummary(client, filepath.Base(file), text)
		if err != nil {
			return "", nil, "", err
		}

		summaries = append(summaries, summary)
		combinedText.WriteString(fmt.Sprintf("\n\n--- Article: %s ---\n\n%s", filepath.Base(file), text))
	}

	// Get group summary
	groupSummary, err := getGroupSummary(client, subject, combinedText.String())
	if err != nil {
		return "", nil, "", err
	}

	return subject, summaries, groupSummary, nil
}

func getArticleSummary(client *openai.Client, filename, text string) (ArticleSummary, error) {
	chunks := splitText(text, maxTextLength)
	var summary ArticleSummary

	for _, chunk := range chunks {
		resp, err := client.CreateChatCompletion(
			context.Background(),
			openai.ChatCompletionRequest{
				Model: openai.GPT4,
				Messages: []openai.ChatCompletionMessage{
					{
						Role: openai.ChatMessageRoleSystem,
						Content: `Extract the following information as JSON:
{
	"title": "[Article title]",
	"objectives": "[Main objectives]",
	"study_type": "[Study type]",
	"methodology": "[Methodology]",
	"findings": "[Main findings]",
	"conclusions": "[Conclusions]",
	"limitations": "[Limitations]",
	"keywords": ["keyword1", "keyword2"]
}`,
					},
					{
						Role:    openai.ChatMessageRoleUser,
						Content: fmt.Sprintf("Filename: %s\n\nContent:\n%s", filename, chunk),
					},
				},
				Temperature: 0.3, // Lower temp for more factual responses
			},
		)

		if err != nil {
			return summary, fmt.Errorf("API error: %v", err)
		}

		// Parse the JSON response
		var partial ArticleSummary
		if err := json.Unmarshal([]byte(resp.Choices[0].Message.Content), &partial); err != nil {
			return summary, fmt.Errorf("error parsing summary: %v", err)
		}

		// Merge partial summaries
		summary = mergeSummaries(summary, partial)
	}

	return summary, nil
}

func getGroupSummary(client *openai.Client, subject, text string) (string, error) {
	resp, err := client.CreateChatCompletion(
		context.Background(),
		openai.ChatCompletionRequest{
			Model: openai.GPT4,
			Messages: []openai.ChatCompletionMessage{
				{
					Role: openai.ChatMessageRoleSystem,
					Content: fmt.Sprintf(`Provide a comprehensive summary of key themes, common methodologies, 
and overall findings across these articles about %s. Highlight any contrasting viewpoints 
or particularly notable findings.`, subject),
				},
				{
					Role:    openai.ChatMessageRoleUser,
					Content: text,
				},
			},
			Temperature: 0.5,
		},
	)

	if err != nil {
		return "", fmt.Errorf("API error: %v", err)
	}

	return resp.Choices[0].Message.Content, nil
}

// ==============================
// HELPER FUNCTIONS
// ==============================
func determineSubject(files []string) string {
	// Simple heuristic - use common prefix of filenames
	if len(files) == 0 {
		return "General"
	}

	commonPrefix := files[0]
	for _, file := range files[1:] {
		commonPrefix = commonPrefix[:commonPrefixLength(commonPrefix, file)]
	}

	// Clean up the prefix
	subject := strings.TrimSuffix(filepath.Base(commonPrefix), "-_ ")
	if subject == "" {
		return "General"
	}
	return subject
}

func commonPrefixLength(a, b string) int {
	i := 0
	for ; i < len(a) && i < len(b) && a[i] == b[i]; i++ {
	}
	return i
}

func extractTextFromPDF(path string) (string, error) {
	doc, err := fitz.New(path)
	if err != nil {
		return "", err
	}
	defer doc.Close()

	var textBuilder strings.Builder
	for i := 0; i < doc.NumPage(); i++ {
		pageText, err := doc.Text(i)
		if err != nil {
			return "", err
		}
		textBuilder.WriteString(pageText + "\n\n")
	}

	return textBuilder.String(), nil
}

func splitText(text string, maxLength int) []string {
	var chunks []string
	for len(text) > maxLength {
		splitAt := strings.LastIndex(text[:maxLength], "\n\n")
		if splitAt == -1 {
			splitAt = maxLength
		}
		chunks = append(chunks, text[:splitAt])
		text = text[splitAt:]
	}
	if len(text) > 0 {
		chunks = append(chunks, text)
	}
	return chunks
}

func mergeSummaries(a, b ArticleSummary) ArticleSummary {
	if a.Title == "" {
		a.Title = b.Title
	}
	if a.Objectives == "" {
		a.Objectives = b.Objectives
	}
	if a.StudyType == "" {
		a.StudyType = b.StudyType
	}
	if a.Methodology == "" {
		a.Methodology = b.Methodology
	}
	if a.Findings == "" {
		a.Findings = b.Findings
	}
	if a.Conclusions == "" {
		a.Conclusions = b.Conclusions
	}
	if a.Limitations == "" {
		a.Limitations = b.Limitations
	}

	// Merge keywords without duplicates
	keywordMap := make(map[string]bool)
	for _, kw := range a.Keywords {
		keywordMap[kw] = true
	}
	for _, kw := range b.Keywords {
		if !keywordMap[kw] {
			a.Keywords = append(a.Keywords, kw)
			keywordMap[kw] = true
		}
	}

	return a
}

func saveStructuredResults(results Results) error {
	file, err := os.Create(resultsFile)
	if err != nil {
		return err
	}
	defer file.Close()

	encoder := json.NewEncoder(file)
	encoder.SetIndent("", "  ")
	return encoder.Encode(results)
}

func getPDFFiles(dir string) ([]string, error) {
	var pdfFiles []string

	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, err
	}

	for _, entry := range entries {
		if !entry.IsDir() && strings.ToLower(filepath.Ext(entry.Name())) == ".pdf" {
			pdfFiles = append(pdfFiles, filepath.Join(dir, entry.Name()))
		}
	}

	return pdfFiles, nil
}

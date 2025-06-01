#!/usr/bin/env bun
// innovation-curator-agent.ts - Streamlined innovation entity curation using MCP

import { Command } from "commander";
import { config } from "dotenv";
import path from "path";
import fs from "fs";
import { readFile, writeFile, appendFile } from "fs/promises";
import chalk from "chalk";
import { z } from "zod";
import { generateText, tool } from "ai";
import { experimental_createMCPClient as createMCPClient } from "ai";
import cliProgress from "cli-progress";
import { createAzure } from "@ai-sdk/azure";

// Load environment variables
config();

// -------------------- Global State --------------------
let resultsTable: ResultTableRow[] = [];
let currentPairIndex = 0;
let totalPairs = 0;
let mcpClient: any = null;
let mcpTools: any = null;
let outputJsonlPath = "";

// -------------------- Interfaces --------------------
interface ResultTableRow {
  index: number;
  innovation1: string;
  innovation2: string;
  decision: string;
  status: string;
  statusCode: string;
  canonical_id: string;
  confidence: number;
}

interface CurationResult {
  resolutions: any[];
}

// -------------------- Directory Setup --------------------
const DATA_DIR = path.join(process.cwd(), "data");
if (!fs.existsSync(DATA_DIR)) fs.mkdirSync(DATA_DIR);

const RESULTS_DIR = path.join(DATA_DIR, "results");
if (!fs.existsSync(RESULTS_DIR)) fs.mkdirSync(RESULTS_DIR);

const CURATED_DIR = path.join(DATA_DIR, "curated");
if (!fs.existsSync(CURATED_DIR)) fs.mkdirSync(CURATED_DIR);

// -------------------- Azure OpenAI Configuration --------------------
const AZURE_CONFIG_PATH = path.join(DATA_DIR, "keys", "azure_config.json");

interface AzureConfig {
  api_key: string;
  api_version: string;
  deployment: string;
  resource_name?: string;
}

function loadAzureConfig(): AzureConfig {
  try {
    const config = JSON.parse(fs.readFileSync(AZURE_CONFIG_PATH, "utf-8"));
    return config["gpt-4.1-mini"];
  } catch (error) {
    console.error(
      chalk.red(`Failed to load Azure config from ${AZURE_CONFIG_PATH}`)
    );
    throw error;
  }
}

// -------------------- Helper Functions --------------------
function getStatusColor(statusCode: string): (text: string) => string {
  switch (statusCode) {
    case "merged":
    case "canonical_created":
      return chalk.green;
    case "kept_separate":
    case "both_processed":
      return chalk.blue;
    case "duplicate_rejected":
      return chalk.yellow;
    case "invalid":
    case "error":
      return chalk.red;
    default:
      return chalk.white;
  }
}

// Initialize MCP client
async function initializeMcpClient() {
  try {
    mcpClient = await createMCPClient({
      transport: {
        type: "sse",
        url: "http://localhost:9000/sse",
      },
    });

    console.log(chalk.green("✓ MCP client initialized successfully"));
    mcpTools = await mcpClient.tools();

    console.log(chalk.cyan("Available MCP tools:"));
    const toolNames = Object.keys(mcpTools);
    toolNames.forEach((name) => {
      console.log(chalk.cyan(`  - ${name}`));
    });

    return mcpTools;
  } catch (error) {
    console.error(chalk.red(`Failed to initialize MCP client: ${error}`));
    throw error;
  }
}

// Cleanup MCP client
async function cleanupMcpClient() {
  if (mcpClient) {
    try {
      await mcpClient.close();
      console.log(chalk.green("✓ MCP client closed successfully"));
    } catch (error) {
      console.error(chalk.yellow(`Error closing MCP client: ${error}`));
    }
  }
}

// -------------------- JSONL Helper --------------------
async function appendToJsonl(data: any): Promise<void> {
  try {
    const jsonLine = JSON.stringify(data) + "\n";
    await appendFile(outputJsonlPath, jsonLine, "utf-8");
  } catch (error: any) {
    console.error(chalk.red(`Error appending to JSONL: ${error.message}`));
  }
}

// -------------------- Streamlined Tool ---------------------
const recordDuplicateDecision = tool({
  description: "Record the final duplicate decision and update result tracking",
  parameters: z.object({
    innovation1_name: z.string().describe("First innovation name"),
    innovation2_name: z.string().describe("Second innovation name"),

    // Decision details
    are_same_innovation: z
      .boolean()
      .describe("Whether these represent the same innovation"),
    canonical_innovation_id: z
      .string()
      .describe("Canonical innovation ID created or used"),
    canonical_innovation_name: z.string().describe("Canonical innovation name"),

    // Processing details
    decision_reasoning: z.string().describe("Reasoning for the decision"),
    status: z
      .string()
      .describe("Status of the operation (merged/kept_separate/error)"),
    confidence: z
      .number()
      .min(0)
      .max(1)
      .describe("Confidence in this decision"),
  }),
  execute: async (params): Promise<string> => {
    let statusCode = params.status;
    let statusText = params.status;

    // Format status text for display
    if (params.are_same_innovation) {
      statusText = "Merged to Canonical";
      statusCode = "merged";
    } else {
      statusText = "Kept Separate";
      statusCode = "kept_separate";
    }

    resultsTable.push({
      index: currentPairIndex,
      innovation1: params.innovation1_name,
      innovation2: params.innovation2_name,
      decision: params.are_same_innovation ? "SAME" : "DIFFERENT",
      status: statusText,
      statusCode: statusCode,
      canonical_id: params.canonical_innovation_id,
      confidence: params.confidence,
    });

    // Append to JSONL immediately
    const jsonlEntry = {
      canonical_innovation_id: params.canonical_innovation_id,
      canonical_name: params.canonical_innovation_name,
      decision: params.are_same_innovation ? "SAME" : "DIFFERENT",
      status: statusText,
      status_code: statusCode,
      confidence: params.confidence,
      reasoning: params.decision_reasoning,
      decision_timestamp: new Date().toISOString(),
      pair_index: currentPairIndex,
      innovation1_name: params.innovation1_name,
      innovation2_name: params.innovation2_name,
    };

    await appendToJsonl(jsonlEntry);

    return JSON.stringify(
      {
        success: true,
        pair_index: currentPairIndex,
        innovation1: params.innovation1_name,
        innovation2: params.innovation2_name,
        decision: params.are_same_innovation ? "SAME" : "DIFFERENT",
        canonical_id: params.canonical_innovation_id,
        canonical_name: params.canonical_innovation_name,
        reasoning: params.decision_reasoning,
        status: statusText,
        status_code: statusCode,
      },
      null,
      2
    );
  },
});

// -------------------- Processing Functions --------------------
async function processDuplicatePair(
  duplicatePair: any,
  pairIndex: number,
  total: number,
  progressBar: cliProgress.SingleBar
): Promise<any> {
  currentPairIndex = pairIndex;

  const innovation1 = duplicatePair.innovation1;
  const innovation2 = duplicatePair.innovation2;

  // Update progress bar with current pair info
  progressBar.update(pairIndex, {
    status: `${innovation1.name.substring(
      0,
      25
    )}... vs ${innovation2.name.substring(0, 25)}...`,
  });

  // Create detailed comparison context
  const prompt = `
    Analyze these two innovation mentions to determine if they represent the SAME innovation or DIFFERENT innovations:
    
    ═══════════════════════════════════════════════════════════════════════════════
    INNOVATION 1:
    ═══════════════════════════════════════════════════════════════════════════════
    Name: "${innovation1.name}"
    Description: "${innovation1.description}"
    Source URL: ${innovation1.source_url}
    Document ID: ${innovation1.source_doc_id}
    Dataset: ${innovation1.dataset_origin}
    Associated Organizations: ${innovation1.associated_orgs.join(", ")}
    Non-VTT Organizations: ${innovation1.non_vtt_orgs.join(", ")}
    Publication Date: ${innovation1.publication_date}
    Document Type: ${innovation1.document_type}
    
    ═══════════════════════════════════════════════════════════════════════════════
    INNOVATION 2:
    ═══════════════════════════════════════════════════════════════════════════════
    Name: "${innovation2.name}"
    Description: "${innovation2.description}"
    Source URL: ${innovation2.source_url}
    Document ID: ${innovation2.source_doc_id}
    Dataset: ${innovation2.dataset_origin}
    Associated Organizations: ${innovation2.associated_orgs.join(", ")}
    Non-VTT Organizations: ${innovation2.non_vtt_orgs.join(", ")}
    Publication Date: ${innovation2.publication_date}
    Document Type: ${innovation2.document_type}
    
    ═══════════════════════════════════════════════════════════════════════════════
    SIMILARITY ANALYSIS:
    ═══════════════════════════════════════════════════════════════════════════════
    Similarity Score: ${duplicatePair.similarity_score}
    Shared Organizations: ${duplicatePair.shared_non_vtt_orgs.join(", ")}
    Confidence Level: ${duplicatePair.confidence_level}
    Match Reasons: ${duplicatePair.match_reasons.join(", ")}
    URL Match: ${duplicatePair.url_match}
    Domain Match: ${duplicatePair.domain_match}
    
    You are processing duplicate pair ${pairIndex} of ${total}. Complete the following steps:
    
    1. **ANALYZE CAREFULLY**: Compare the innovation names, descriptions, organizations, and context
    2. **MAKE DECISION**: Determine if they represent the SAME innovation or DIFFERENT innovations
    3. **IF SAME**: 
       - Use resolve_or_create_canonical_innovation to create a unified innovation
       - Use the better description and combine relevant details
       - Process BOTH innovation's organizational relationships
    4. **IF DIFFERENT**:
       - Use resolve_or_create_canonical_innovation for EACH innovation separately
       - Process each innovation's organizational relationships independently
    5. **FOR EACH RESOLVED INNOVATION**:
       - Use resolve_or_create_organization for each associated org (VAT ID)
       - Use add_mention_to_link to create INVOLVED_IN relationships
    6. **RECORD DECISION**: Use recordDuplicateDecision to track your choice
    
    **DECISION CRITERIA:**
    - Same core technology/innovation concept = SAME
    - Different applications of similar tech = DIFFERENT  
    - Language variations (EN/FI) of same innovation = SAME
    - Different projects/phases of same tech = POTENTIALLY SAME
    - Completely different innovations = DIFFERENT
    
    **BE THOROUGH**: Process ALL organizational relationships for the canonical innovation(s).
    
    Work step by step, showing your reasoning throughout.
  `;

  try {
    const azureConfig = loadAzureConfig();

    // Set environment variables for Azure configuration
    process.env.AZURE_API_KEY = azureConfig.api_key;
    process.env.AZURE_RESOURCE_NAME =
      azureConfig.resource_name || "aaltoaihack25-resource";
    process.env.AZURE_API_VERSION = azureConfig.api_version;

    // Create Azure model instance with correct usage
    const azureModel = createAzure({
      resourceName: azureConfig.resource_name || "aaltoaihack25-resource",
      apiKey: azureConfig.api_key,
      apiVersion: azureConfig.api_version,
    })(azureConfig.deployment);

    // Use generateText instead of streamText
    const result = await generateText({
      model: azureModel,
      tools: {
        ...mcpTools,
        recordDuplicateDecision,
      },
      maxSteps: 25, // Increased for more complex processing
      onStepFinish: ({ toolCalls, toolResults, stepType }) => {
        // Update progress bar with current tool being executed
        if (toolCalls && toolCalls.length > 0) {
          const currentTool = toolCalls[toolCalls.length - 1];
          if (currentTool) {
            progressBar.update(pairIndex, {
              status: `${currentTool.toolName} (${pairIndex}/${total})`,
            });
          }
        }
      },
      system: `
      You are the Innovation Duplicate Curator for the VTT Innovation Knowledge Graph. Your job is to determine whether candidate duplicate pairs represent the same innovation or different innovations, then create the appropriate canonical entities and relationships.
      
      # Your Core Responsibilities
      1. Make accurate same/different decisions for innovation pairs
      2. Create canonical innovation entities (avoiding duplication)
      3. Process ALL organizational relationships for each canonical innovation
      4. Maintain full provenance through mention records
      5. Use the MCP tools systematically to build the clean graph
      
      # Decision Guidelines
      
      ## SAME Innovation Indicators:
      - Core technology/method is identical
      - Same fundamental innovation with language variations (EN/FI)
      - Same project with different descriptions/perspectives
      - Same organizations developing same technology
      - Clear semantic equivalence despite naming differences
      
      ## DIFFERENT Innovation Indicators:  
      - Fundamentally different technologies/methods
      - Different applications (even if similar domain)
      - Different organizational leads/contexts
      - Clearly distinct innovations that happen to be similar
      
      ## Examples:
      
      ### SAME: Language Variations
      "Solar Foods protein production technology" (EN)
      "Ruokaa ilmasta uudella prosessilla" (FI - "Food from air with new process")
      → Both describe the same CO2-to-protein technology
      
      Example tool usage:
      \`\`\`typescript
      // First innovation
      resolve_or_create_canonical_innovation(
        "Solar Foods protein production technology",
        "Technology that produces protein from air using CO2",
        {
          source_doc_id: "EN_123",
          dataset_origin: "vtt_domain",
          associated_orgs: ["FI12345678"],
          non_vtt_orgs: ["FI12345678"],
          source_url: "https://example.com/solar-foods",
          publication_date: "2023-01-01",
          document_type: "VTT_Domain",
          company_name: "Solar Foods",
          has_vtt_involvement: true
        }
      )

      // Second innovation (same as first, different language)
      resolve_or_create_canonical_innovation(
        "Ruokaa ilmasta uudella prosessilla",
        "Teknologia, joka tuottaa proteiinia ilmasta CO2:sta",
        {
          source_doc_id: "FI_456",
          dataset_origin: "vtt_domain",
          associated_orgs: ["FI12345678"],
          non_vtt_orgs: ["FI12345678"],
          source_url: "https://example.com/ruokaa-ilmasta",
          publication_date: "2023-01-02",
          document_type: "VTT_Domain",
          company_name: "Solar Foods",
          has_vtt_involvement: true
        }
      )
      \`\`\`
      
      ### SAME: Naming Variations  
      "Neo-Carbon Food protein production process"
      "Solar Foods protein production technology"  
      → Both refer to the same collaborative VTT+LUT protein-from-air innovation
      
      ### DIFFERENT: Similar but Distinct
      "CRISPR gene editing for crops"
      "CRISPR gene editing for cancer treatment"
      → Same base technology, different applications = DIFFERENT innovations
      
      # Processing Workflow
      
      For SAME innovations:
      1. Create ONE canonical innovation with combined/best description
      2. Process organizational relationships from BOTH mentions
      3. Ensure all organizations get proper INVOLVED_IN relationships
      
      For DIFFERENT innovations:
      1. Create separate canonical innovation for each
      2. Process organizational relationships for each independently
      3. Maintain distinct innovation identities
      
      # MCP Tool Usage Pattern
      
      ## For Each Innovation Entity:
      1. resolve_or_create_canonical_innovation(name, description, mention_context)
         - name: The innovation name
         - description: The innovation description
         - mention_context: Object containing:
           - source_doc_id: Document ID where innovation was found
           - dataset_origin: Dataset source
           - associated_orgs: List of associated organization VAT IDs
           - non_vtt_orgs: List of non-VTT organization VAT IDs
           - source_url: URL where innovation was found
           - publication_date: When the innovation was published
           - document_type: Type of document (e.g. VTT_Domain)
           - company_name: Name of company mentioned
           - has_vtt_involvement: Whether VTT is involved
      2. For each associated organization:
         - resolve_or_create_organization(vat_id, name_hint)  
         - add_mention_to_link(org_vat_id, innovation_id, mention_record)
      
      ## Always End With:
      recordDuplicateDecision(decision_details)
      
      **CRITICAL**: Process ALL organizational relationships. Don't skip any associated_orgs!
      `,
      prompt,
    });

    // Process the results from generateText
    let finalResult = null;
    let decisionResult = null;

    // Extract canonical innovation result from tool results
    for (const toolResult of result.toolResults) {
      if (toolResult.toolName === "resolve_or_create_canonical_innovation") {
        finalResult =
          typeof toolResult.result === "string"
            ? JSON.parse(toolResult.result)
            : toolResult.result;
      }

      if (toolResult.toolName === "recordDuplicateDecision") {
        decisionResult =
          typeof toolResult.result === "string"
            ? JSON.parse(toolResult.result)
            : toolResult.result;

        const statusColor = getStatusColor(decisionResult.status_code);
        console.log(
          `   ${chalk.gray(`[${pairIndex}/${total}]`)} ${statusColor(
            decisionResult.decision
          )} → ${chalk.cyan(decisionResult.canonical_name.substring(0, 60))}...`
        );
      }
    }

    // If no decision was recorded in tool results, check if it was recorded globally
    if (!decisionResult && resultsTable.length > 0) {
      const lastResult = resultsTable[resultsTable.length - 1];
      if (lastResult && lastResult.index === pairIndex) {
        const statusColor = getStatusColor(lastResult.statusCode);
        console.log(
          `   ${chalk.gray(`[${pairIndex}/${total}]`)} ${statusColor(
            lastResult.decision
          )} → ${chalk.cyan(lastResult.canonical_id.substring(0, 60))}...`
        );
      }
    }

    return {
      ...(finalResult || {
        success: true,
        innovation1: innovation1.name,
        innovation2: innovation2.name,
        pair_index: pairIndex,
      }),
      tool_results: result.toolResults,
      decision_result: decisionResult,
    };
  } catch (error: any) {
    console.log(
      `   ${chalk.gray(`[${pairIndex}/${total}]`)} ${chalk.red("ERROR")} → ${
        error.message
      }`
    );

    resultsTable.push({
      index: pairIndex,
      innovation1: innovation1.name,
      innovation2: innovation2.name,
      decision: "ERROR",
      status: "Error",
      statusCode: "error",
      canonical_id: "error",
      confidence: 0,
    });

    // Also append error to JSONL
    const errorEntry = {
      canonical_innovation_id: "error",
      canonical_name: "ERROR",
      decision: "ERROR",
      status: "Error",
      status_code: "error",
      confidence: 0,
      reasoning: error.message,
      decision_timestamp: new Date().toISOString(),
      pair_index: pairIndex,
      innovation1_name: innovation1.name,
      innovation2_name: innovation2.name,
    };
    await appendToJsonl(errorEntry);

    return {
      success: false,
      error: error.message,
      innovation1: innovation1.name,
      innovation2: innovation2.name,
    };
  }
}

// Main function to process all duplicate pairs
async function processDuplicatePairs(
  duplicatesData: any[]
): Promise<CurationResult> {
  resultsTable = [];
  totalPairs = duplicatesData.length;

  await initializeMcpClient();

  console.log(
    chalk.bold.blue("╔═════════════════════════════════════════════════╗")
  );
  console.log(
    chalk.bold.blue("║       INNOVATION DUPLICATE CURATOR v2           ║")
  );
  console.log(
    chalk.bold.blue("╚═════════════════════════════════════════════════╝")
  );

  console.log(
    chalk.green(
      `\n🔍 Processing ${totalPairs} duplicate candidate pairs using MCP tools...\n`
    )
  );

  // Initialize empty JSONL file
  await writeFile(outputJsonlPath, "", "utf-8");
  console.log(chalk.cyan(`📝 Initialized JSONL output: ${outputJsonlPath}`));

  // Simple, clean progress bar
  const progressBar = new cliProgress.SingleBar(
    {
      format:
        "Progress |{bar}| {percentage}% | {value}/{total} pairs | {status}",
      barCompleteChar: "█",
      barIncompleteChar: "░",
      hideCursor: true,
    },
    cliProgress.Presets.shades_classic
  );

  progressBar.start(totalPairs, 0, { status: "Starting..." });

  try {
    const resolutions = [];

    for (let i = 0; i < duplicatesData.length; i++) {
      const duplicatePair = duplicatesData[i];
      const pairIndex = i + 1;

      const result = await processDuplicatePair(
        duplicatePair,
        pairIndex,
        totalPairs,
        progressBar
      );

      if (result) {
        resolutions.push({
          ...result,
          pair_info: {
            innovation1: duplicatePair.innovation1,
            innovation2: duplicatePair.innovation2,
            similarity_score: duplicatePair.similarity_score,
            confidence_level: duplicatePair.confidence_level,
          },
        });
      }
    }

    progressBar.stop();
    console.log(); // Clean line after progress bar

    printResultsTable();
    printSummaryStatistics();
    await cleanupMcpClient();

    return { resolutions };
  } catch (error: any) {
    progressBar.stop();
    console.error(chalk.red.bold(`Error in processing: ${error.message}`));
    await cleanupMcpClient();
    throw error;
  }
}

// Print results table
function printResultsTable(): void {
  console.log(chalk.bold("\n📊 Duplicate Curation Results:"));
  console.log(chalk.dim("─".repeat(140)));
  console.log(
    chalk.bold(
      "#   | Innovation 1 vs Innovation 2                    | Decision  | Status          | Canonical ID      | Conf"
    )
  );
  console.log(chalk.dim("─".repeat(140)));

  for (const result of resultsTable) {
    const statusColor = getStatusColor(result.statusCode);
    const decisionColor =
      result.decision === "SAME"
        ? chalk.green
        : result.decision === "DIFFERENT"
        ? chalk.blue
        : chalk.red;

    console.log(
      `${String(result.index).padEnd(3)} | ${(
        result.innovation1.substring(0, 20) +
        " vs " +
        result.innovation2.substring(0, 20)
      ).padEnd(48)} | ${decisionColor(
        result.decision.padEnd(9)
      )} | ${statusColor(result.status.padEnd(15))} | ${result.canonical_id
        .substring(0, 18)
        .padEnd(18)} | ${result.confidence.toFixed(2)}`
    );
  }

  console.log(chalk.dim("─".repeat(140)));
}

// Print summary statistics
function printSummaryStatistics(): void {
  const statusCounts: Record<string, number> = {};
  const decisionCounts: Record<string, number> = {};
  let validCount = 0;

  for (const result of resultsTable) {
    statusCounts[result.statusCode] =
      (statusCounts[result.statusCode] || 0) + 1;
    decisionCounts[result.decision] =
      (decisionCounts[result.decision] || 0) + 1;

    if (result.statusCode !== "error") {
      validCount++;
    }
  }

  const sameCount = decisionCounts.SAME || 0;
  const differentCount = decisionCounts.DIFFERENT || 0;
  const errorCount = decisionCounts.ERROR || 0;
  const totalCount = resultsTable.length || 1;

  console.log(
    chalk.bold(
      `\n🔍 Valid pairs processed: ${validCount}/${resultsTable.length} (${(
        (validCount / totalCount) *
        100
      ).toFixed(1)}%)`
    )
  );
  console.log(
    chalk.bold(
      `🔗 Same innovation (merged): ${sameCount} (${(
        (sameCount / totalCount) *
        100
      ).toFixed(1)}%)`
    )
  );
  console.log(
    chalk.bold(
      `🔀 Different innovations (kept separate): ${differentCount} (${(
        (differentCount / totalCount) *
        100
      ).toFixed(1)}%)`
    )
  );
  console.log(
    chalk.bold(
      `❌ Processing errors: ${errorCount} (${(
        (errorCount / totalCount) *
        100
      ).toFixed(1)}%)`
    )
  );

  // Calculate deduplication impact
  const potentialReduction = sameCount; // Each "same" decision removes one duplicate
  console.log(
    chalk.bold(`🎯 Potential duplicates eliminated: ${potentialReduction}`)
  );
}

// -------------------- File Processing Function --------------------
async function processFile(filePath: string): Promise<void> {
  try {
    const content = await readFile(filePath, "utf-8");
    const data = JSON.parse(content);

    if (!data.duplicates || !Array.isArray(data.duplicates)) {
      throw new Error(
        `Invalid data format in file: ${filePath}. Expected {"duplicates": [...]}`
      );
    }

    console.log(
      chalk.cyan(
        `📊 File contains ${data.total_raw_mentions} raw mentions, ${data.unique_innovations} unique innovations, ${data.duplicates.length} duplicate pairs`
      )
    );

    // Set up output paths
    const baseFileName = path.basename(filePath, ".json");
    const outputJsonPath = path.join(
      CURATED_DIR,
      `${baseFileName}_curated.json`
    );
    outputJsonlPath = path.join(
      RESULTS_DIR,
      `${baseFileName}_duplicate_decisions.jsonl`
    );

    const result = await processDuplicatePairs(data.duplicates);

    await writeFile(outputJsonPath, JSON.stringify(result, null, 2), "utf-8");
    console.log(chalk.green(`💾 JSON result saved to: ${outputJsonPath}`));
    console.log(chalk.green(`💾 JSONL decisions saved to: ${outputJsonlPath}`));
  } catch (error: any) {
    console.error(
      chalk.red(`Error processing file ${filePath}: ${error.message}`)
    );
    throw error;
  }
}

// Main function
async function main() {
  const program = new Command();

  program
    .name("innovation-curator-agent")
    .description("Streamlined innovation entity curation using MCP")
    .version("1.0.0")
    .argument("<input>", "Input JSON file containing duplicate pairs")
    .action(async (input: string) => {
      try {
        const stats = fs.statSync(input);
        if (stats.isDirectory()) {
          console.error(
            chalk.red.bold(
              "Error: Directory input is not supported. Please provide a single JSON file."
            )
          );
          process.exit(1);
        }

        if (!input.endsWith(".json")) {
          console.error(chalk.red.bold("Error: Input must be a JSON file."));
          process.exit(1);
        }

        await processFile(input);
      } catch (error: any) {
        console.error(chalk.red(`Error: ${error.message}`));
        process.exit(1);
      }
    });

  program.parse(process.argv);
}

main();

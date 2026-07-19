#import <AppKit/AppKit.h>
#import <Foundation/Foundation.h>
#import <WebKit/WebKit.h>

@interface BookPDFRenderer : NSObject <WKNavigationDelegate>

@property(nonatomic, strong) NSURL *inputURL;
@property(nonatomic, strong) NSURL *outputURL;
@property(nonatomic, strong) NSURL *metadataURL;
@property(nonatomic, strong) NSString *preparationScript;
@property(nonatomic, strong) WKWebView *webView;
@property(nonatomic, strong) NSWindow *window;
@property(nonatomic, strong) dispatch_block_t timeoutBlock;
@property(nonatomic) CGFloat contentHeight;

- (instancetype)initWithInputURL:(NSURL *)inputURL
                       outputURL:(NSURL *)outputURL
                      metadataURL:(NSURL *)metadataURL
                preparationScript:(NSString *)preparationScript;
- (void)start;

@end

@implementation BookPDFRenderer

- (instancetype)initWithInputURL:(NSURL *)inputURL
                       outputURL:(NSURL *)outputURL
                      metadataURL:(NSURL *)metadataURL
                preparationScript:(NSString *)preparationScript {
  self = [super init];
  if (self) {
    _inputURL = inputURL;
    _outputURL = outputURL;
    _metadataURL = metadataURL;
    _preparationScript = preparationScript;

    WKWebViewConfiguration *configuration =
        [[WKWebViewConfiguration alloc] init];
    configuration.websiteDataStore = [WKWebsiteDataStore nonPersistentDataStore];
    _webView = [[WKWebView alloc]
        initWithFrame:NSMakeRect(0, 0, 794, 1123)
        configuration:configuration];
    _webView.navigationDelegate = self;
  }
  return self;
}

- (void)start {
  self.window = [[NSWindow alloc]
      initWithContentRect:self.webView.frame
                styleMask:NSWindowStyleMaskBorderless
                  backing:NSBackingStoreBuffered
                    defer:NO];
  self.window.contentView = self.webView;
  [self.window orderOut:nil];

  __weak BookPDFRenderer *weakSelf = self;
  self.timeoutBlock = dispatch_block_create(0, ^{
    [weakSelf finishWithError:@"WebKit render timed out"];
  });
  dispatch_after(dispatch_time(DISPATCH_TIME_NOW, 300 * NSEC_PER_SEC),
                 dispatch_get_main_queue(), self.timeoutBlock);

  [self.webView loadFileURL:self.inputURL
      allowingReadAccessToURL:[self.inputURL URLByDeletingLastPathComponent]];
}

- (void)webView:(WKWebView *)webView
    didFinishNavigation:(WKNavigation *)navigation {
  NSLog(@"Navigation finished; waiting for MathJax and fonts");
  NSString *readinessScript = @
      "if (window.MathJax?.startup?.promise) {"
      "  await Promise.race(["
      "    window.MathJax.startup.promise,"
      "    new Promise((_, reject) => setTimeout("
      "      () => reject(new Error('MathJax timeout')), 60000"
      "    ))"
      "  ]);"
      "}"
      "if (document.fonts?.ready) {"
      "  await document.fonts.ready;"
      "}"
      "await new Promise(resolve => setTimeout(resolve, 100));";
  NSString *script = [readinessScript
      stringByAppendingString:self.preparationScript];

  __weak BookPDFRenderer *weakSelf = self;
  [webView callAsyncJavaScript:script
                    arguments:@{}
                       inFrame:nil
                inContentWorld:[WKContentWorld pageWorld]
              completionHandler:^(id result, NSError *error) {
    if (error != nil) {
      [weakSelf finishWithError:
          [NSString stringWithFormat:@"JavaScript readiness check failed: %@",
                                     error]];
      return;
    }
    if ([result isKindOfClass:[NSDictionary class]]) {
      NSDictionary *layout = (NSDictionary *)result;
      NSNumber *height = layout[@"height"];
      weakSelf.contentHeight = height.doubleValue;
      NSLog(@"Web content ready: height=%@, MathJax=%@, units=%lu, "
            "keep ranges=%lu",
            height, layout[@"mathContainers"],
            (unsigned long)[layout[@"unitStarts"] count],
            (unsigned long)[layout[@"keepRanges"] count]);
      NSError *jsonError = nil;
      NSData *metadata = [NSJSONSerialization
          dataWithJSONObject:result
                     options:NSJSONWritingPrettyPrinted | NSJSONWritingSortedKeys
                       error:&jsonError];
      if (metadata == nil ||
          ![metadata writeToURL:weakSelf.metadataURL
                        options:NSDataWritingAtomic
                          error:&jsonError]) {
        [weakSelf finishWithError:
            [NSString stringWithFormat:@"Metadata write failed: %@", jsonError]];
        return;
      }
    }
    dispatch_after(dispatch_time(DISPATCH_TIME_NOW, 500 * NSEC_PER_MSEC),
                   dispatch_get_main_queue(), ^{
      [weakSelf printPDF];
    });
  }];
}

- (void)webView:(WKWebView *)webView
    didFailNavigation:(WKNavigation *)navigation
            withError:(NSError *)error {
  [self finishWithError:
      [NSString stringWithFormat:@"Navigation failed: %@", error]];
}

- (void)webView:(WKWebView *)webView
    didFailProvisionalNavigation:(WKNavigation *)navigation
                       withError:(NSError *)error {
  [self finishWithError:
      [NSString stringWithFormat:@"Provisional navigation failed: %@", error]];
}

- (void)printPDF {
  if (self.contentHeight <= 0) {
    [self finishWithError:@"Web content height is unavailable"];
    return;
  }

  WKPDFConfiguration *configuration = [[WKPDFConfiguration alloc] init];
  configuration.rect = CGRectMake(0, 0, self.webView.bounds.size.width,
                                  self.contentHeight);

  __weak BookPDFRenderer *weakSelf = self;
  [self.webView createPDFWithConfiguration:configuration
                         completionHandler:^(NSData *data, NSError *error) {
    if (error != nil || data == nil) {
      [weakSelf finishWithError:
          [NSString stringWithFormat:@"WebKit PDF creation failed: %@", error]];
      return;
    }
    NSError *writeError = nil;
    if (![data writeToURL:weakSelf.outputURL
                  options:NSDataWritingAtomic
                    error:&writeError]) {
      [weakSelf finishWithError:
          [NSString stringWithFormat:@"PDF write failed: %@", writeError]];
      return;
    }
    if (weakSelf.timeoutBlock != nil) {
      dispatch_block_cancel(weakSelf.timeoutBlock);
    }
    NSLog(@"WebKit PDF written to %@ (%lu bytes)", weakSelf.outputURL.path,
          (unsigned long)data.length);
    [[NSApplication sharedApplication] terminate:nil];
  }];
}

- (void)finishWithError:(NSString *)message {
  if (self.timeoutBlock != nil) {
    dispatch_block_cancel(self.timeoutBlock);
  }
  fprintf(stderr, "error: %s\n", message.UTF8String);
  exit(1);
}

@end

int main(int argc, const char *argv[]) {
  @autoreleasepool {
    if (argc != 5) {
      fprintf(stderr,
              "usage: render_pdf_webkit <input.html> <output.pdf> "
              "<metadata.json> <prepare.js>\n");
      return 2;
    }

    NSURL *inputURL =
        [NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]];
    NSURL *outputURL =
        [NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[2]]];
    NSURL *metadataURL =
        [NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[3]]];
    NSString *scriptPath =
        [NSString stringWithUTF8String:argv[4]];
    NSError *scriptError = nil;
    NSString *preparationScript =
        [NSString stringWithContentsOfFile:scriptPath
                                  encoding:NSUTF8StringEncoding
                                     error:&scriptError];
    if (preparationScript == nil) {
      fprintf(stderr, "error: cannot read preparation script: %s\n",
              scriptError.localizedDescription.UTF8String);
      return 2;
    }

    NSApplication *application = [NSApplication sharedApplication];
    [application setActivationPolicy:NSApplicationActivationPolicyProhibited];

    BookPDFRenderer *renderer =
        [[BookPDFRenderer alloc] initWithInputURL:inputURL
                                       outputURL:outputURL
                                      metadataURL:metadataURL
                                preparationScript:preparationScript];
    [renderer start];
    [application run];
  }
  return 0;
}
